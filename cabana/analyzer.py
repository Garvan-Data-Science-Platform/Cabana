import cv2
import time
import numpy as np
from .log import Log
import networkx as nx
from numba import jit
import imageio.v3 as iio
import scipy.ndimage as ndi
from .detector import FibreDetector
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize, remove_small_holes


class SkeletonAnalyzer:
    """
    Analyzes the morphology of skeletonized structures in binary images.

    This class provides functionality to detect and analyze skeletonized structures,
    calculating metrics such as total length, branch points, end points, fractal dimension,
    lacunarity, and curvature measures.

    Attributes:
        skel_thresh (int): Threshold for minimum size of skeleton segments to keep
        branch_thresh (int): Threshold for minimum branch length to keep
        hole_thresh (int): Threshold for hole size in preprocessing
        dark_line (bool): If True, process dark lines on light background
        FOREGROUND (int): Pixel value representing foreground elements (default: 255)
        BACKGROUND (int): Pixel value representing background elements (default: 0)
    """

    def __init__(self, skel_thresh=20, branch_thresh=10, hole_threshold=8, dark_line=True):
        """
        Initialize the SkeletonAnalyzer with analysis parameters.

        Args:
            skel_thresh (int): Threshold for minimum size of skeleton segments (pixels)
            branch_thresh (int): Threshold for minimum branch length (pixels)
            hole_threshold (int): Threshold for hole size in preprocessing
            dark_line (bool): If True, process dark lines on light background
        """
        self.pts_image = None
        self.skel_thresh = skel_thresh
        self.branch_thresh = branch_thresh
        self.hole_thresh = hole_threshold

        # Metrics
        self.proj_area = 0.0
        self.num_tips = 0
        self.num_branches = 0
        self.total_length = 0.0
        self.growth_unit = 0.0
        self.frac_dim = 0.0
        self.lacunarity = 0.0
        self.avg_curve_long = 0.0
        self.avg_curve_all = 0.0
        self.avg_curve_spline = 0.0

        # Images
        self.raw_image = None
        self.skel_image = None
        self.pruned_image = None
        self.key_pts_image = None
        self.long_path_image = None

        # Maps
        self.curve_map_long = None
        self.curve_map_all = None
        self.length_map_long = None
        self.length_map_all = None

        # Graph representation
        self.subgraphs = []

        # Line detection settings
        self.dark_line = dark_line
        self.FOREGROUND = 255
        self.BACKGROUND = 0

    def reset(self):
        """Reset all analysis results and intermediate data."""
        # Reset metrics
        self.proj_area = 0.0
        self.num_tips = 0
        self.num_branches = 0
        self.total_length = 0.0
        self.growth_unit = 0.0
        self.frac_dim = 0.0
        self.lacunarity = 0.0
        self.avg_curve_long = 0.0
        self.avg_curve_all = 0.0
        self.avg_curve_spline = 0.0

        # Reset images
        self.raw_image = None
        self.skel_image = None
        self.pruned_image = None
        self.key_pts_image = None
        self.long_path_image = None

        # Reset maps
        self.curve_map_long = None
        self.curve_map_all = None
        self.length_map_long = None
        self.length_map_all = None

        # Reset graph representation
        self.subgraphs = []

    @staticmethod
    @jit(nopython=True)
    def count_neighbors(skel_image, y, x, radius=1, val=1):
        """
        Count neighboring pixels with specified value.

        Args:
            skel_image (ndarray): Skeleton image
            y (int): Y-coordinate of the pixel
            x (int): X-coordinate of the pixel
            radius (int): Neighborhood radius
            val (int): Value to count

        Returns:
            int: Number of neighbors with the specified value
        """
        count = 0
        for i in range(y - radius, y + radius + 1):
            for j in range(x - radius, x + radius + 1):
                if skel_image[i, j] == val and ((y != i) or (x != j)):
                    count += 1
        return count

    @staticmethod
    def traverse_skeletons(skel_image, end_points, brh_points, foreground):
        """
        Traverse the skeleton to identify and measure paths between key points.

        This method builds a graph representation of the skeleton by traversing paths
        between end points and branch points, calculating the length of each path.

        Args:
            skel_image (ndarray): Binary skeleton image
            end_points (list): List of (y, x) coordinates of skeleton end points
            brh_points (list): List of (y, x) coordinates of branch points
            foreground (int): Pixel value representing the foreground

        Returns:
            list: List of tuples (src, dst, length, path, type) for each path
        """
        # Initialize a list to store lengths and paths
        lengths_paths = []
        visited = set()

        # Handle special cases based on point availability
        if not end_points and not brh_points:
            return lengths_paths
        elif end_points and not brh_points:  # only end points available
            if len(end_points) == 1:
                Log.logger.warning("Isolated point found. Ignored!")
                return lengths_paths
            elif len(end_points) > 2:
                Log.logger.warning("A branch with more than 2 points was ignored!")
                return lengths_paths
            else:
                # Traverse from first end point to the second
                stack = [(end_points[0], 0, [end_points[0]])]
                while stack:
                    current, length, path = stack.pop()

                    if current == end_points[1]:
                        lengths_paths.append((path[0], current, length, path, 'end-to-end'))
                        continue  # Skip adding neighbors if destination is reached

                    visited.add(current)

                    for neighbor in SkeletonAnalyzer.get_neighbors(skel_image, current, foreground):
                        if neighbor not in visited:
                            dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                           (neighbor[1] - current[1]) ** 2.0)
                            new_path = path + [neighbor]
                            stack.append((neighbor, length + dist, new_path))

        elif brh_points and not end_points:  # only branch points available
            # Trace from branch points to branch points
            for src in brh_points:
                # The last element in the tuple indicates if another point has been visited
                stack = [(src, 0, [src], False)]

                while stack:
                    current, length, path, visited_other = stack.pop()

                    if current == src and visited_other:
                        # Found a self-loop
                        lengths_paths.append((src, src, length, path, 'brh-to-brh'))
                        continue

                    if current in visited:
                        continue  # Avoid reprocessing nodes except for the self-loop check

                    visited.add(current)
                    neighbors = SkeletonAnalyzer.get_neighbors(skel_image, current, foreground)
                    is_reached = [(n != src) and (n in brh_points) for n in neighbors]
                    if any(is_reached):
                        neighbor = neighbors[np.where(is_reached)[0][0]]
                        dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                       (neighbor[1] - current[1]) ** 2.0)
                        new_path = path + [neighbor]
                        lengths_paths.append((path[0], neighbor, length + dist, new_path, 'brh-to-brh'))
                        continue

                    for neighbor in neighbors:
                        if neighbor not in visited:
                            dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                           (neighbor[1] - current[1]) ** 2.0)
                            new_path = path + [neighbor]
                            stack.append(
                                (neighbor, length + dist, new_path, True if neighbor != src else visited_other))

        else:  # both end points and branch points exist
            # First trace from end points to branch points
            stack = [(src, 0, [src]) for src in end_points]
            while stack:
                current, length, path = stack.pop()

                if current in brh_points:
                    lengths_paths.append((path[0], current, length, path, 'end-to-brh'))
                    continue

                visited.add(current)

                # If any neighbors is in destination points
                neighbors = SkeletonAnalyzer.get_neighbors(skel_image, current, foreground)
                neighbors_reached = [n for n in neighbors if n in brh_points]
                if neighbors_reached:
                    neighbor = neighbors_reached[0]
                    dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                   (neighbor[1] - current[1]) ** 2.0)
                    new_path = path + [neighbor]
                    lengths_paths.append((path[0], neighbor, length + dist, new_path, 'end-to-brh'))
                    continue

                for neighbor in neighbors:
                    if neighbor not in visited:
                        dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                       (neighbor[1] - current[1]) ** 2.0)
                        new_path = path + [neighbor]
                        stack.append((neighbor, length + dist, new_path))

            # Then trace from branch points to branch points
            for src in brh_points:
                # The last element in the tuple indicates if another point has been visited
                stack = [(src, 0, [src], False)]

                while stack:
                    current, length, path, visited_other = stack.pop()

                    if current == src and visited_other:
                        # Found a self-loop
                        lengths_paths.append((src, current, length, path, 'brh-to-brh'))
                        continue

                    if current in visited:
                        continue

                    visited.add(current)
                    neighbors = SkeletonAnalyzer.get_neighbors(skel_image, current, foreground)
                    is_reached = [(n != src) and (n in brh_points) for n in neighbors]
                    if any(is_reached):
                        neighbor = neighbors[np.where(is_reached)[0][0]]
                        dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                       (neighbor[1] - current[1]) ** 2.0)
                        new_path = path + [neighbor]
                        lengths_paths.append((src, neighbor, length + dist, new_path, 'brh-to-brh'))
                        continue

                    for neighbor in neighbors:
                        if neighbor not in visited:
                            dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                           (neighbor[1] - current[1]) ** 2.0)
                            new_path = path + [neighbor]
                            stack.append(
                                (neighbor, length + dist, new_path, True if neighbor != src else visited_other))

        return lengths_paths

    @staticmethod
    @jit(nopython=True)
    def get_neighbors(skel_image, point, foreground):
        """
        Get neighboring foreground points of a given point.

        Args:
            skel_image (ndarray): Binary skeleton image
            point (tuple): (row, col) coordinates of the point
            foreground (int): Pixel value representing the foreground

        Returns:
            list: List of neighboring foreground points as (row, col) tuples
        """
        row, col = point
        neighbors = []
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if (i != row or j != col) and 0 <= i < skel_image.shape[0] and 0 <= j < skel_image.shape[1]:
                    if skel_image[i, j] == foreground:
                        neighbors.append((i, j))
        return neighbors

    @staticmethod
    @jit(nopython=True)
    def is_branchpoint(skel_image, row, col, foreground):
        """
        Determine if a pixel is a branch point based on its neighborhood pattern.

        Args:
            skel_image (ndarray): Binary skeleton image
            row (int): Row index of the pixel
            col (int): Column index of the pixel
            foreground (int): Pixel value representing the foreground

        Returns:
            bool: True if the pixel is a branch point, False otherwise
        """
        height, width = skel_image.shape[:2]
        # Search north and south
        for y in [row - 1, row + 1]:
            previous = -1
            for x in range(col - 1, col + 2):
                if x < 0 or x >= width or y < 0 or y >= height:
                    break
                if skel_image[y, x] == foreground and previous == foreground:
                    return False
                previous = skel_image[y, x]

        # Search east and west
        for x in [col - 1, col + 1]:
            previous = -1
            for y in range(row - 1, row + 2):
                if x < 0 or x >= width or y < 0 or y >= height:
                    break
                if skel_image[y, x] == foreground and previous == foreground:
                    return False
                previous = skel_image[y, x]

        return True

    @staticmethod
    def longest_path(graph, wgt='length'):
        """
        Find the longest path in a graph.

        Uses Dijkstra's algorithm to find the pair of nodes with the maximum
        weighted distance between them, then reconstructs the path.

        Args:
            graph (nx.Graph): NetworkX graph
            wgt (str): Edge attribute to use as weight

        Returns:
            tuple: (list of nodes in the longest path, length of the path)
        """
        # First, get all shortest path lengths using Dijkstra's algorithm
        all_distances = dict(nx.all_pairs_dijkstra_path_length(graph, weight=wgt))

        # Initialize variables to keep track of the longest path found
        max_length = 0
        max_path_nodes = (None, None)

        # Iterate over all pairs and find the one with the greatest distance
        for source, target_dict in all_distances.items():
            for target, distance in target_dict.items():
                if distance > max_length:
                    max_length = distance
                    max_path_nodes = (source, target)

        # Extract the longest path using the nodes found
        if max_path_nodes[0] is not None and max_path_nodes[1] is not None:
            # We use nx.dijkstra_path to get the path itself
            longest_path = nx.dijkstra_path(graph, max_path_nodes[0], max_path_nodes[1], weight=wgt)
            return longest_path, max_length
        else:
            return [], 0

    @staticmethod
    @jit(nopython=True)
    def is_endpoint(skel_image, row, col, background):
        """
        Determine if a pixel is an endpoint based on the longest chain of background pixels.

        Args:
            skel_image (ndarray): Binary skeleton image
            row (int): Row index of the pixel
            col (int): Column index of the pixel
            background (int): Pixel value representing the background

        Returns:
            bool: True if the pixel is an endpoint, False otherwise
        """
        # Define the 8-connected neighborhood relative positions
        neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        longest_chain = 0
        current_chain = 0

        # Start checking from the pixel's right neighbor and move in a circular manner
        for i in range(len(neighbors) * 2):  # Multiply by 2 to allow wrapping around the neighborhood
            dx, dy = neighbors[i % len(neighbors)]
            ny, nx = row + dy, col + dx

            # Check if the neighbor is within the image bounds
            if 0 <= ny < skel_image.shape[0] and 0 <= nx < skel_image.shape[1]:
                if skel_image[ny, nx] == background:  # Background pixel
                    current_chain += 1
                    longest_chain = max(longest_chain, current_chain)
                else:  # Foreground pixel, reset chain length
                    current_chain = 0
            else:
                # Treat out-of-bounds as background to continue the chain
                current_chain += 1

        # If the longest chain of background pixels is 5 or more, it's considered an endpoint
        return longest_chain >= 5

    def construct_graphs(self):
        """
        Construct graph representations of the skeleton.

        This method:
        1. Labels connected components in the skeleton
        2. Removes small segments below threshold
        3. Identifies end points and branch points
        4. Builds graph representation with nodes and edges
        5. Prunes short branches

        The result is stored in self.subgraphs as a list of NetworkX graphs.
        """
        # Label each connected component
        eight_con = np.ones((3, 3), dtype=int)
        labels, num = ndi.label(self.skel_image, eight_con)

        # Remove small skeletons below threshold
        if self.skel_thresh > 0:
            segment_counts = ndi.sum(self.skel_image > 0, labels, range(1, num + 1))
            labels_remove = np.where(segment_counts <= self.skel_thresh)[0]

            for label in labels_remove:
                self.skel_image[np.where(labels == label + 1)] = self.BACKGROUND

            # Relabel after deleting short skeletons
            labels, num = ndi.label(self.skel_image, eight_con)

        # Count the number of neighbors for each pixel
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        num_neighbors = ndi.convolve((self.skel_image == self.FOREGROUND).astype(np.uint8),
                                     kernel, mode="constant")

        height, width = self.skel_image.shape[:2]

        # Helper function to get 3x3 neighborhood as binary matrix
        def get_binary_3x3(img, row, col):
            i1 = False if row - 1 < 0 or col - 1 < 0 else img[row - 1, col - 1] == self.FOREGROUND
            i2 = False if row - 1 < 0 else img[row - 1, col] == self.FOREGROUND
            i3 = False if row - 1 < 0 or col + 1 >= width else img[row - 1, col + 1] == self.FOREGROUND
            i4 = False if col - 1 < 0 else img[row, col - 1] == self.FOREGROUND
            i5 = True  # Center pixel (always foreground in this context)
            i6 = False if col + 1 >= width else img[row, col + 1] == self.FOREGROUND
            i7 = False if row + 1 >= height or col - 1 < 0 else img[row + 1, col - 1] == self.FOREGROUND
            i8 = False if row + 1 >= height else img[row + 1, col] == self.FOREGROUND
            i9 = False if row + 1 >= height or col + 1 >= width else img[row + 1, col + 1] == self.FOREGROUND
            return np.array([[i1, i2, i3], [i4, i5, i6], [i7, i8, i9]])

        # Define structural elements for endpoint detection
        selems_2 = list()
        selems_2.append(np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]]))
        selems_2.append(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]]))
        selems_2 = [np.rot90(selems_2[i], k=j) for i in range(2) for j in range(4)]

        # Define structural elements for branch point detection with 3 neighbors
        selems_3 = list()
        selems_3.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
        selems_3.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
        selems_3.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
        selems_3.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
        selems_3.append(np.array([[0, 0, 1], [1, 1, 0], [0, 0, 1]]))
        selems_3 = [np.rot90(selems_3[i], k=j) for i in range(5) for j in range(4)]

        # Define structural elements for branch point detection with 4 neighbors
        selems_4 = selems_3.copy()
        selems_4.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        selems_4.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
        selems_tmp = list()
        selems_tmp.append(np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]]))
        selems_tmp.append(np.array([[0, 1, 0], [0, 1, 1], [1, 1, 0]]))
        selems_tmp.append(np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]]))
        selems_4.extend([np.rot90(selems_tmp[i], k=j) for i in range(3) for j in range(4)])

        self.pruned_image = self.skel_image.copy()

        # Process each connected component
        for label in range(1, num + 1):
            # Canvas image to draw a segment at a time
            canvas = np.zeros(self.skel_image.shape, dtype=np.uint8)
            row_idx, col_idx = np.where(labels == label)
            canvas[row_idx, col_idx] = self.FOREGROUND
            row_start, row_end = row_idx.min(), row_idx.max()
            col_start, col_end = col_idx.min(), col_idx.max()

            # Find endpoints and branch points
            endpoints, branchpoints = [], []
            for row in range(row_start, row_end + 1):
                for col in range(col_start, col_end + 1):
                    if canvas[row, col] == self.FOREGROUND:
                        if num_neighbors[row, col] == 1:
                            # Pixel with only one neighbor is definitely an endpoint
                            endpoints.append((row, col))
                        elif num_neighbors[row, col] == 2:
                            # Check if this is a special case endpoint with 2 neighbors
                            binary = get_binary_3x3(self.skel_image, row, col)
                            for selem in selems_2:
                                if not np.logical_xor(binary, selem).any():
                                    endpoints.append((row, col))
                                    break
                        elif num_neighbors[row, col] == 3:
                            # Check if this is a branch point with 3 neighbors
                            binary = get_binary_3x3(self.skel_image, row, col)
                            for selem in selems_3:
                                if not np.logical_xor(binary, selem).any():
                                    branchpoints.append((row, col))
                                    break
                        elif num_neighbors[row, col] == 4:
                            # Check if this is a branch point with 4 neighbors
                            binary = get_binary_3x3(self.skel_image, row, col)
                            for selem in selems_4:
                                if not np.logical_xor(binary, selem).any():
                                    branchpoints.append((row, col))
                                    break
                        elif num_neighbors[row, col] > 4:
                            # Any pixel with more than 4 neighbors is a branch point
                            branchpoints.append((row, col))

            # Create a graph for this component
            G = nx.Graph()

            # Add nodes to the graph
            for end_pt in endpoints:
                G.add_node(end_pt, node_type="end-point")
            for brh_pt in branchpoints:
                G.add_node(brh_pt, node_type="branch-point")

            # Find paths between key points and add edges
            lengths_paths = SkeletonAnalyzer.traverse_skeletons(canvas, endpoints, branchpoints, self.FOREGROUND)
            for src, dst, length, path, typ in lengths_paths:
                G.add_edge(src, dst, length=length, path=path, type=typ)

            # Prune short branches
            edges_to_remove = [edge for edge in G.edges(data=True)
                               if edge[2]['length'] <= self.branch_thresh and
                               edge[2]['type'] == 'end-to-brh']
            G.remove_edges_from([edge[:2] for edge in edges_to_remove])

            # Remove isolated nodes
            isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
            G.remove_nodes_from(isolated_nodes)

            # Add the graph to the list of subgraphs
            self.subgraphs.append(G)

            # Update pruned image by removing pruned branches
            for edge in edges_to_remove:
                path_array = np.asarray(edge[2]['path'])[:-1, :]
                self.pruned_image[path_array[:, 0], path_array[:, 1]] = self.BACKGROUND

            for node in isolated_nodes:
                self.pruned_image[node[0], node[1]] = self.BACKGROUND

    def calc_curve_all(self, win_sz=11):
        """
        Calculate curvature for all points in the pruned skeleton.

        Uses a sliding window to estimate the local curvature at each point by
        measuring the angle between line segments.

        Args:
            win_sz (int): Window size for curvature calculation. Should be odd.

        The results are stored in self.curve_map_all and self.avg_curve_all.
        """
        curve_map = np.zeros_like(self.pruned_image, dtype=float)
        count_map = np.zeros_like(self.pruned_image, dtype=int)
        side = win_sz // 2

        # Process each edge in all subgraphs
        for G in self.subgraphs:
            for u, v, a in G.edges(data=True):
                edge_path = a['path']
                if len(edge_path) > 0:
                    # Pad the path to handle edge points
                    points = np.pad(np.asarray(edge_path), ((side, side), (0, 0)),
                                    mode='reflect', reflect_type='odd')

                    # Calculate curvature at each point along the path
                    for j in range(side, len(points) - side):
                        # Calculate angles of line segments before and after the point
                        theta1 = np.arctan2(
                            points[j - side, 0] - points[j, 0], points[j, 1] - points[j - side, 1])
                        theta2 = np.arctan2(
                            points[j, 0] - points[j + side, 0], points[j + side, 1] - points[j, 1])

                        # Calculate the angle difference (curvature)
                        angle_diff = abs(theta1 - theta2)
                        if angle_diff >= np.pi:
                            angle_diff = 2 * np.pi - angle_diff

                        # Store the curvature value and increment the count
                        curve_map[points[j, 0], points[j, 1]] += np.rad2deg(angle_diff)
                        count_map[points[j, 0], points[j, 1]] += 1

        # Calculate average curvature at each point
        calc_mask = count_map >= 1
        curve_map[calc_mask] /= count_map[calc_mask]

        self.curve_map_all = curve_map
        self.avg_curve_all = np.mean(curve_map[calc_mask]) if calc_mask.any() else 0.0

    def calc_curve_long(self, win_sz=11):
        """
        Calculate curvature along the longest path in each subgraph.

        Args:
            win_sz (int): Window size for curvature calculation. Should be odd.

        The results are stored in self.curve_map_long and self.avg_curve_long.
        """
        self.curve_map_long = np.zeros_like(self.pruned_image, dtype=float)
        side = win_sz // 2
        curvatures = []

        # Process each subgraph
        for G in self.subgraphs:
            # Find the longest path in this subgraph
            longest_path = SkeletonAnalyzer.longest_path(G, "length")[0]
            trajectory = []

            if longest_path:  # If a path was found
                src_node = longest_path[0]
                # Reconstruct the complete path by connecting segments
                for mid_idx, dst_node in enumerate(longest_path[1:]):
                    edge_path = G.edges[src_node, dst_node]['path']

                    if mid_idx == 0:
                        # For the first segment, add the entire path
                        trajectory.extend(edge_path)
                    else:
                        # For subsequent segments, avoid duplicating the connection point
                        if trajectory[-1] == edge_path[0]:
                            trajectory.extend(edge_path[1:])
                        elif trajectory[-1] == edge_path[-1]:
                            # If the path is reversed, add it in reverse order
                            trajectory.extend(edge_path[::-1][1:])
                        else:
                            # There's a discontinuity in the path
                            pass

                    src_node = dst_node

            if trajectory:
                # Pad the path to handle edge points
                points = np.pad(np.asarray(trajectory), ((side, side), (0, 0)),
                                mode='reflect', reflect_type='odd')

                # Calculate curvature at each point along the longest path
                for j in range(side, len(points) - side):
                    # Calculate angles of line segments before and after the point
                    theta1 = np.arctan2(
                        points[j - side, 0] - points[j, 0], points[j, 1] - points[j - side, 1])
                    theta2 = np.arctan2(
                        points[j, 0] - points[j + side, 0], points[j + side, 1] - points[j, 1])

                    # Calculate the angle difference (curvature)
                    angle_diff = abs(theta1 - theta2)
                    if angle_diff >= np.pi:
                        angle_diff = 2 * np.pi - angle_diff

                    # Store the curvature value
                    self.curve_map_long[points[j, 0], points[j, 1]] = np.rad2deg(angle_diff)
                    curvatures.append(np.rad2deg(angle_diff))

        # Calculate the average curvature across all longest paths
        self.avg_curve_long = np.mean(curvatures) if curvatures else 0.0

    def points_test(self):
        """
        Test function to visually identify and count end points and branch points.

        This function analyzes the skeleton image to find endpoints and branch points
        based on their local neighborhood patterns. It creates a visualization with:
        - End points marked in green (0, 255, 0)
        - Branch points marked in yellow (255, 255, 0)

        The function counts both types of points and logs the results.
        The visualization is stored in self.pts_image.
        """

        # Helper function to get 3x3 neighborhood as binary matrix
        def get_binary_3x3(img, row, col):
            """Extract a 3x3 binary neighborhood around a point as a boolean matrix."""
            i1 = False if row - 1 < 0 or col - 1 < 0 else img[row - 1, col - 1] == self.FOREGROUND
            i2 = False if row - 1 < 0 else img[row - 1, col] == self.FOREGROUND
            i3 = False if row - 1 < 0 or col + 1 >= width else img[row - 1, col + 1] == self.FOREGROUND
            i4 = False if col - 1 < 0 else img[row, col - 1] == self.FOREGROUND
            i5 = True  # Center pixel is always foreground
            i6 = False if col + 1 >= width else img[row, col + 1] == self.FOREGROUND
            i7 = False if row + 1 >= height or col - 1 < 0 else img[row + 1, col - 1] == self.FOREGROUND
            i8 = False if row + 1 >= height else img[row + 1, col] == self.FOREGROUND
            i9 = False if row + 1 >= height or col + 1 >= width else img[row + 1, col + 1] == self.FOREGROUND
            return np.array([[i1, i2, i3], [i4, i5, i6], [i7, i8, i9]])

        # Kernel for counting neighbors
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

        # Count the number of neighbors for each pixel
        num_neighbors = ndi.convolve((self.pruned_image == self.FOREGROUND).astype(np.uint8),
                                     kernel, mode="constant")

        # Create a color image for visualization
        self.pts_image = np.repeat(self.pruned_image[:, :, None], 3, axis=2)
        height, width = self.pruned_image.shape[:2]
        brh_pts_cnt = 0  # Branch points counter
        end_pts_cnt = 0  # End points counter

        # Define structural elements for endpoint detection (2 neighbors)
        selems_2 = list()
        selems_2.append(np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]]))  # Corner pattern 1
        selems_2.append(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]]))  # Corner pattern 2
        # Generate rotations of the structural elements for all possible orientations
        selems_2 = [np.rot90(selems_2[i], k=j) for i in range(2) for j in range(4)]

        # Define structural elements for branch point detection (3 neighbors)
        selems_3 = list()
        selems_3.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))  # T-junction pattern
        selems_3.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))  # Diagonal pattern 1
        selems_3.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))  # Diagonal pattern 2
        selems_3.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))  # Corner pattern 1
        selems_3.append(np.array([[0, 0, 1], [1, 1, 0], [0, 0, 1]]))  # Corner pattern 2
        # Generate rotations for all possible orientations
        selems_3 = [np.rot90(selems_3[i], k=j) for i in range(5) for j in range(4)]

        # Define structural elements for branch point detection (4 neighbors)
        selems_4 = selems_3.copy()
        selems_4.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))  # Cross pattern
        selems_4.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))  # X pattern
        selems_tmp = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]])  # Special corner pattern
        selems_4.extend([np.rot90(selems_tmp, k=j) for j in range(4)])

        # Scan all pixels and identify end points and branch points
        for row in range(height):
            for col in range(width):
                if self.skel_image[row, col] == self.FOREGROUND:
                    # Case 1: Pixel with only 1 neighbor is definitely an endpoint
                    if num_neighbors[row, col] == 1:
                        cv2.circle(self.pts_image, (col, row), 3, (0, 255, 0), 2)  # Green circle for endpoint
                        end_pts_cnt += 1
                    # Case 2: Special patterns with 2 neighbors that are endpoints
                    elif num_neighbors[row, col] == 2:
                        binary = get_binary_3x3(self.skel_image, row, col)
                        for selem in selems_2:
                            # If the pattern matches exactly (no logical differences)
                            if not np.logical_xor(binary, selem).any():
                                cv2.circle(self.pts_image, (col, row), 3, (0, 255, 0), 2)  # Green circle for endpoint
                                end_pts_cnt += 1
                                break
                    # Case 3: Branch points with 3 neighbors in specific patterns
                    elif num_neighbors[row, col] == 3:
                        binary = get_binary_3x3(self.skel_image, row, col)
                        for selem in selems_3:
                            if not np.logical_xor(binary, selem).any():
                                cv2.circle(self.pts_image, (col, row), 3, (255, 255, 0),
                                           2)  # Yellow circle for branch point
                                brh_pts_cnt += 1
                                break
                    # Case 4: Branch points with 4 neighbors in specific patterns
                    elif num_neighbors[row, col] == 4:
                        binary = get_binary_3x3(self.skel_image, row, col)
                        for selem in selems_4:
                            if not np.logical_xor(binary, selem).any():
                                cv2.circle(self.pts_image, (col, row), 3, (255, 255, 0),
                                           2)  # Yellow circle for branch point
                                brh_pts_cnt += 1
                                break
                    # Case 5: Any pixel with more than 4 neighbors is definitely a branch point
                    elif num_neighbors[row, col] > 4:
                        cv2.circle(self.pts_image, (col, row), 3, (255, 255, 0), 2)  # Yellow circle for branch point
                        brh_pts_cnt += 1

        # Log the results
        Log.logger.info(f"points_test: {end_pts_cnt} end points, {brh_pts_cnt} branch points.")

    def draw_key_points(self):
        """
        Draw and count key points (end points and branch points) from the graph representation.

        This function identifies key points based on the node degree in the graph:
        - End points (degree 1): Colored red (255, 0, 0)
        - Branch points (degree > 2): Colored yellow (255, 255, 0)

        The counts are stored in self.num_tips and self.num_branches.
        The visualization is stored in self.key_pts_image.
        """
        # Create a color image for visualization
        self.key_pts_image = np.repeat(self.pruned_image[:, :, None], 3, axis=2)
        brh_pts_cnt = 0  # Branch points counter
        end_pts_cnt = 0  # End points counter

        # Process each subgraph
        for G in self.subgraphs:
            for node, attrs in G.nodes(data=True):
                # End points have only one connection (degree 1)
                if G.degree(node) == 1:
                    cv2.circle(self.key_pts_image, (node[1], node[0]), 3, (255, 0, 0), 2)  # Red circle for endpoint
                    end_pts_cnt += 1
                # Branch points have more than two connections (degree > 2)
                elif G.degree(node) > 2:
                    cv2.circle(self.key_pts_image, (node[1], node[0]), 3, (255, 255, 0),
                               2)  # Yellow circle for branch point
                    brh_pts_cnt += 1
                # Nodes with degree 2 are just path points - no special marking

        # Store the counts as class attributes
        self.num_tips = end_pts_cnt
        self.num_branches = brh_pts_cnt

        # The original print statement is commented out
        # print(f"draw_key_points: {end_pts_cnt} end points, {brh_pts_cnt} branch points.")

    def calc_total_len(self):
        """
        Calculate the total length of the skeleton.

        This is a simple count of foreground pixels in the pruned image.
        The result is stored in self.total_length.
        """
        self.total_length = np.sum(self.pruned_image == self.FOREGROUND)

    def calc_growth_unit(self):
        """
        Calculate the growth unit metric.

        Growth unit is defined as 2.0 * total_length / (num_tips + num_branches).
        This represents the average length between significant points in the structure.

        The result is stored in self.growth_unit.
        """
        # Avoid division by zero
        if self.num_tips + self.num_branches > 0:
            self.growth_unit = 2.0 * self.total_length / (self.num_tips + self.num_branches)
        else:
            self.growth_unit = 0.0

    def calc_len_map_all(self):
        """
        Calculate a length map for all paths in the skeleton.

        For each point in each path, store the total length of the path it belongs to.
        The result is stored in self.length_map_all.
        """
        self.length_map_all = np.zeros_like(self.pruned_image, dtype=float)

        # Process each edge in all subgraphs
        for G in self.subgraphs:
            for u, v, a in G.edges(data=True):
                edge_path = a['path']
                if len(edge_path) > 0:
                    # Convert to numpy array for vector operations
                    points = np.asarray(edge_path)

                    # Calculate the Euclidean distance between consecutive points
                    traj_len = np.sum(
                        np.sqrt((points[1:, 0] - points[:-1, 0]) ** 2 +
                                (points[1:, 1] - points[:-1, 1]) ** 2))

                    # Assign the total path length to each point in the path
                    self.length_map_all[points[:, 0], points[:, 1]] = traj_len

    def calc_len_map_long(self):
        """
        Calculate a length map for the longest path in each subgraph.

        For each point in the longest path, store the total length of that path.
        The result is stored in self.length_map_long.
        """
        self.length_map_long = np.zeros_like(self.pruned_image, dtype=float)

        # Process each subgraph
        for G in self.subgraphs:
            # Find the longest path in this subgraph
            longest_path = SkeletonAnalyzer.longest_path(G, "length")[0]
            trajectory = []

            if longest_path:  # If a path was found
                src_node = longest_path[0]
                # Reconstruct the complete path by connecting segments
                for mid_idx, dst_node in enumerate(longest_path[1:]):
                    edge_path = G.edges[src_node, dst_node]['path']

                    if mid_idx == 0:
                        # For the first segment, add the entire path
                        trajectory.extend(edge_path)
                    else:
                        # For subsequent segments, avoid duplicating the connection point
                        if trajectory[-1] == edge_path[0]:
                            trajectory.extend(edge_path[1:])
                        elif trajectory[-1] == edge_path[-1]:
                            # If the path is reversed, add it in reverse order
                            trajectory.extend(edge_path[::-1][1:])
                        else:
                            # There's a discontinuity in the path
                            pass

                    src_node = dst_node

            if trajectory:
                # Convert to numpy array for vector operations
                points = np.asarray(trajectory)

                # Calculate the Euclidean distance between consecutive points
                traj_len = np.sum(np.sqrt((points[1:, 0] - points[:-1, 0]) ** 2 +
                                          (points[1:, 1] - points[:-1, 1]) ** 2))

                # Assign the total path length to each point in the path
                self.length_map_long[points[:, 0], points[:, 1]] = traj_len

    def draw_longest_path(self):
        """
        Draw the longest path in each subgraph.

        Creates a visualization where the longest path in each subgraph
        is highlighted in yellow on the pruned image.

        The result is stored in self.long_path_image.
        """
        # Create a color image for visualization
        self.long_path_image = np.repeat(self.pruned_image[:, :, None], 3, axis=2)

        # Process each subgraph
        for G in self.subgraphs:
            # Find the longest path in this subgraph
            longest_path = SkeletonAnalyzer.longest_path(G, "length")[0]

            if longest_path:  # If a path was found
                src_node = longest_path[0]
                # Draw each segment of the longest path
                for dst_node in longest_path[1:]:
                    # Get the path between source and destination nodes
                    edge_path = np.asarray(G.edges[src_node, dst_node]['path'])

                    # Draw the path as a yellow polyline
                    # Note: OpenCV's polylines expects points in (x,y) format, while our points are (y,x)
                    cv2.polylines(self.long_path_image, [np.asarray(edge_path)[:, ::-1]],
                                  False, [255, 255, 0], 2)

                    # Update source node for the next segment
                    src_node = dst_node

    def calc_frac_dim(self):
        """
        Calculate the fractal dimension of the skeleton.

        Uses box-counting method to estimate the fractal dimension:
        1. Count the number of boxes of different sizes needed to cover the structure
        2. Plot log(count) vs log(size) and measure the slope

        The result is stored in self.frac_dim.
        """
        height, width = self.pruned_image.shape[:2]

        # Minimal dimension of image
        p = min(self.pruned_image.shape)

        # Greatest power of 2 less than or equal to p
        n = 2 ** np.floor(np.log(p) / np.log(2)) - 2

        # Extract the exponent
        n = int(np.log(n) / np.log(2))

        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2 ** np.arange(n, 1, -1)

        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            # Sum the foreground pixels in boxes of a given size
            S = np.add.reduceat(
                np.add.reduceat(self.pruned_image == self.FOREGROUND,
                                np.arange(0, height, size), axis=0),
                np.arange(0, width, size), axis=1)

            # Count boxes that contain foreground pixels but aren't completely filled
            counts.append(len(np.where((S > 0) & (S < size ** 2))[0]))

        # Check if all counts are zero
        if np.all(counts == 0):
            Log.logger.warning("All counts are zero. Fractal dimension cannot be computed meaningfully.")
            return None
        else:
            # Replace zero counts with a small positive number to avoid log(0)
            counts = np.maximum(counts, 1e-10)

        # The fractal dimension is the negative slope of the log-log plot
        self.frac_dim = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]

    def calc_lacunarity(self):
        """
        Calculate the lacunarity of the skeleton.

        Lacunarity is a measure of the "gappiness" or texture of a fractal:
        - Higher values indicate more heterogeneous distribution with larger gaps
        - Lower values indicate more homogeneous, uniform distribution

        Defined as: abs(variance / mean^2 - 1)

        The result is stored in self.lacunarity.
        """
        # Create binary mask where foreground pixels are 1
        mask_img = np.zeros_like(self.pruned_image)
        mask_img[self.pruned_image == self.FOREGROUND] = 1

        # Check if there are any foreground pixels
        if np.count_nonzero(mask_img) == 0:
            Log.logger.warning("No foreground pixels. Lacunarity cannot be computed meaningfully.")
            return None
        else:
            # Calculate lacunarity as abs(variance/mean^2 - 1)
            self.lacunarity = abs(np.var(mask_img.flatten()) / np.mean(mask_img.flatten()) ** 2 - 1.0)

    def calc_curve_spline(self, s=3):
        """
        Calculate curvature using spline interpolation.

        Fits a smoothing spline to each edge path and computes analytical curvature
        using first and second derivatives.

        Args:
            s (float): Smoothing factor for the spline. Higher values produce smoother splines.

        The average curvature is stored in self.avg_curve_spline.
        """
        curve_map = np.zeros_like(self.pruned_image, dtype=float)
        count_map = np.zeros_like(self.pruned_image, dtype=int)

        # Process each edge in all subgraphs
        for G in self.subgraphs:
            for u, v, a in G.edges(data=True):
                edge_path = a['path']
                # Need at least 4 points for cubic spline interpolation
                if len(edge_path) > 3:
                    # Convert path to numpy array
                    contour = np.asarray(edge_path)

                    # Fit a parametric spline to the contour points
                    # s controls the smoothness (higher s = smoother curve)
                    tck, u = splprep(contour.T, s=s)

                    # Evaluate the spline to get points on the curve
                    points = np.asarray(splev(u, tck)).round().astype(int).T

                    # Calculate derivatives for curvature computation
                    # First derivatives (dx/dt, dy/dt)
                    dx, dy = splev(u, tck, der=1)
                    # Second derivatives (d²x/dt², d²y/dt²)
                    ddx, ddy = splev(u, tck, der=2)

                    # Calculate curvature at each point using the formula:
                    # κ = |dx*ddy - dy*ddx| / (dx² + dy²)^(3/2)
                    # Add small epsilon to avoid division by zero
                    curvature = (np.abs(dx * ddy - dy * ddx) /
                                 ((dx ** 2 + dy ** 2 + np.finfo(float).eps) ** 1.5))

                    # Store the curvature values in the map
                    curve_map[points[:, 0], points[:, 1]] = curvature
                    count_map[points[:, 0], points[:, 1]] += 1

        # Calculate average curvature for each point (in case of overlapping paths)
        calc_mask = count_map >= 1
        curve_map[calc_mask] /= count_map[calc_mask]

        # Calculate global average curvature
        self.avg_curve_spline = np.mean(curve_map[calc_mask]) if calc_mask.any() else 0.0

    @staticmethod
    def dilate_color(color_image, mask):
        """
        Dilate a color image based on a binary mask.

        This method creates a dilated version of a color image by extending the colors
        from foreground regions into background regions within a small dilation radius.

        Args:
            color_image (ndarray): RGB or BGR color image
            mask (ndarray): Binary mask where foreground has non-zero values

        Returns:
            ndarray: Dilated color image
        """
        # Get the height and width of the images
        height, width = mask.shape

        # Create a kernel for dilation
        kernel = np.ones((3, 3), np.uint8)

        # Dilate the mask to get the area to fill
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Compute the distance transform to find nearest foreground pixels
        dist_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)

        # Get the coordinates of foreground pixels
        nearest_idx = np.round(np.argwhere(dist_transform == 0))

        # Create an output image initialized with the original color image
        output_image = np.copy(color_image)

        # For each pixel in the dilated mask that is not in the original mask
        for i in range(height):
            for j in range(width):
                if dilated_mask[i, j] != 0 and mask[i, j] == 0:
                    # Find the nearest foreground pixel
                    distances = np.sqrt((nearest_idx[:, 0] - i) ** 2 + (nearest_idx[:, 1] - j) ** 2)
                    nearest_pixel = nearest_idx[np.argmin(distances)]

                    # Copy the color from the nearest foreground pixel
                    output_image[i, j] = color_image[nearest_pixel[0], nearest_pixel[1]]

        return output_image

    def calc_proj_area(self):
        """
        Calculate the projected area of the structure.

        This is simply the count of foreground pixels in the original binary image.
        The result is stored in self.proj_area.
        """
        self.proj_area = np.sum(self.raw_image == self.FOREGROUND)

    def analyze_image(self, image):
        """
        Perform comprehensive analysis of a binary image.

        This is the main method that orchestrates the entire analysis workflow:
        1. Load and preprocess the image
        2. Skeletonize the binary structure
        3. Construct graph representation
        4. Calculate various metrics and maps

        Args:
            image: Either a file path string or a numpy array containing the image

        Returns:
            None: Results are stored in the class attributes
        """
        # Load the image if a path is provided, otherwise use the given array
        self.raw_image = iio.imread(image) if isinstance(image, str) else image

        # Convert to grayscale if the image has multiple channels
        if self.raw_image.ndim > 2:
            self.raw_image = self.raw_image[..., 0]

        # Verify the image is binary
        if len(np.unique(self.raw_image.flatten())) > 2:
            Log.logger.warning("Image to be analyzed has to be binary.")
            return

        # Handle dark lines on light background if needed
        if self.dark_line:
            self.raw_image = 255 - self.raw_image
            self.FOREGROUND = self.raw_image.max()
            self.BACKGROUND = self.raw_image.min()

        # Step 1: Skeletonize the binary image
        # First remove small holes, then perform skeletonization
        skeleton = skeletonize(remove_small_holes(self.raw_image == self.FOREGROUND, self.hole_thresh))
        self.skel_image = (skeleton * self.FOREGROUND).astype(np.uint8)

        # Step 2: Construct graph representation
        self.construct_graphs()

        # Step 3: Identify key points (endpoints and branch points)
        self.draw_key_points()

        # Step 4: Calculate length maps
        self.calc_len_map_all()

        # Step 5: Calculate various metrics
        self.calc_total_len()
        self.calc_proj_area()
        self.calc_growth_unit()
        self.calc_frac_dim()
        self.calc_lacunarity()


if __name__ == "__main__":
    t1 = time.time()
    img_path = "~/Downloads/1681222495317.jpg"
    det = FibreDetector(line_widths=[5],
                        low_contrast=100,
                        high_contrast=200,
                        dark_line=False,
                        extend_line=True,
                        correct_pos=False,
                        min_len=5)
    det.detect_lines(img_path)
    print(time.time() - t1)
    det.save_results(save_dir="~/Desktop/img2/", make_binary=True, draw_junc=True, draw_width=True)
    skel = SkeletonAnalyzer(skel_thresh=10, branch_thresh=5)
    skel.analyze_image("~/Desktop/img2/binary_contours.png")
