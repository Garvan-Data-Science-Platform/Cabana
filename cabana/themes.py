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

THEMES = {
    'Dark': THEME_DARK,
    'Light': THEME_LIGHT,
}

DEFAULT_THEME = 'Dark'
