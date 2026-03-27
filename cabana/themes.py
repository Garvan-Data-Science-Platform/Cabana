"""Theme palette definitions for CABANA GUI.

Each theme is a dict mapping color role names to RGBA tuples.
All themes must define the same 20 keys.
"""

THEME_DARK = {
    # Depth layers (darkest → lightest)
    'canvas':           (20, 22, 28),
    'background':       (28, 31, 38),
    'surface':          (38, 42, 52),
    'dock':             (50, 54, 66),
    'elevated':         (62, 67, 80),

    # Interactive state backgrounds
    'hover':            (70, 76, 92),
    'active':           (80, 88, 106),

    # Primary accent — Muted Indigo
    'highlight':        (110, 158, 245),
    'highlight_hover':  (134, 176, 248),
    'highlight_dim':    (80, 122, 206),
    'highlight_subtle': (110, 158, 245, 28),

    # Text hierarchy
    'text':             (220, 225, 232),
    'text_dim':         (138, 145, 158),
    'text_muted':       (75, 82, 98),
    'secondary':        (190, 195, 200),

    # Borders
    'border':           (55, 60, 72),
    'border_subtle':    (42, 46, 56),

    # Semantic
    'success':          (72, 210, 128),
    'warning':          (248, 190, 48),
    'error':            (242, 88, 88),
}

THEME_LIGHT = {
    # Depth layers (lightest → darker for elevated)
    'canvas':           (245, 245, 248),
    'background':       (238, 240, 244),
    'surface':          (246, 247, 250),
    'dock':             (228, 230, 236),
    'elevated':         (255, 255, 255),

    # Interactive state backgrounds
    'hover':            (214, 218, 228),
    'active':           (198, 204, 216),

    # Primary accent — Deep Indigo
    'highlight':        (66, 115, 215),
    'highlight_hover':  (86, 135, 230),
    'highlight_dim':    (50, 95, 185),
    'highlight_subtle': (66, 115, 215, 28),

    # Text hierarchy
    'text':             (32, 36, 44),
    'text_dim':         (100, 106, 120),
    'text_muted':       (168, 174, 186),
    'secondary':        (62, 68, 80),

    # Borders
    'border':           (192, 198, 210),
    'border_subtle':    (210, 214, 222),

    # Semantic
    'success':          (38, 162, 86),
    'warning':          (205, 150, 18),
    'error':            (205, 52, 52),
}

THEME_DARK_WARM = {
    # Depth layers (darkest → lightest, warm undertone)
    'canvas':           (24, 20, 16),
    'background':       (32, 27, 22),
    'surface':          (44, 38, 32),
    'dock':             (58, 50, 42),
    'elevated':         (72, 62, 52),

    # Interactive state backgrounds
    'hover':            (82, 72, 60),
    'active':           (96, 84, 70),

    # Primary accent — Amber
    'highlight':        (235, 168, 52),
    'highlight_hover':  (245, 185, 80),
    'highlight_dim':    (198, 138, 35),
    'highlight_subtle': (235, 168, 52, 28),

    # Text hierarchy
    'text':             (228, 220, 210),
    'text_dim':         (152, 142, 128),
    'text_muted':       (90, 82, 70),
    'secondary':        (200, 192, 180),

    # Borders
    'border':           (68, 58, 48),
    'border_subtle':    (50, 43, 36),

    # Semantic
    'success':          (72, 200, 110),
    'warning':          (248, 190, 48),
    'error':            (230, 78, 68),
}

THEME_DARK_TEAL = {
    # Depth layers (darkest → lightest, cool undertone)
    'canvas':           (16, 22, 26),
    'background':       (22, 30, 36),
    'surface':          (30, 40, 48),
    'dock':             (40, 52, 62),
    'elevated':         (50, 64, 76),

    # Interactive state backgrounds
    'hover':            (58, 74, 88),
    'active':           (68, 86, 102),

    # Primary accent — Teal
    'highlight':        (52, 200, 175),
    'highlight_hover':  (78, 218, 195),
    'highlight_dim':    (36, 168, 145),
    'highlight_subtle': (52, 200, 175, 28),

    # Text hierarchy
    'text':             (215, 225, 230),
    'text_dim':         (128, 145, 155),
    'text_muted':       (68, 82, 92),
    'secondary':        (180, 195, 202),

    # Borders
    'border':           (48, 62, 72),
    'border_subtle':    (36, 47, 55),

    # Semantic
    'success':          (60, 210, 130),
    'warning':          (240, 188, 50),
    'error':            (235, 82, 82),
}

THEMES = {
    'Dark': THEME_DARK,
    'Light': THEME_LIGHT,
    'Dark Warm': THEME_DARK_WARM,
    'Dark Teal': THEME_DARK_TEAL,
}

DEFAULT_THEME = 'Dark'
