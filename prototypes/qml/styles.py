"""Design tokens for the panel prototype: Cabana's four colour themes
(cabana/themes.py: Light and Dark) combined with the card layout."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "cabana_themes", os.path.join(os.path.dirname(__file__), "..", "..", "cabana", "themes.py"))
_themes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_themes)
THEMES = _themes.THEMES          # {'Dark': {...}, 'Light': {...}, 'Dark Warm': ..., 'Dark Teal': ...}


def _hex(rgb, alpha=None):
    if alpha is not None:
        return "#%02X%02X%02X%02X" % ((round(alpha * 255),) + tuple(rgb[:3]))
    return "#%02X%02X%02X" % tuple(rgb[:3])


def _mix(a, b, f):
    """a blended toward b by f (0 = a, 1 = b)."""
    return tuple(round(x + (y - x) * f) for x, y in zip(a[:3], b[:3]))


BASE_LAYOUT = dict(
    size=13, small=11, head=14, rowH=30, rowGap=10, radius=6, btnRadius=8, btnH=34,
    boxRadius=3.5, fieldStyle="filled", sections="cards", cardRadius=10, labels="left",
    labelWidth=118, railOn=True, fluent=False, tonal=False,
    # card anatomy
    cardPad=12, cardBorder=True, cardShadow=False, headerBand=False, footerBand=False,
    footerDivider=False, accentEdge=False, topHighlight=False, titleOutside=False,
    showDesc=False, headerIcons=False, rowLines=False,
    # colour recipes (resolved in tokens())
    cardTint=0.0, borderless=False, ghost=False,
)

# (name, inspiration, overrides). The chosen look: the "Described" card
# (title, one-line description, divider above the action) on a Blender-style
# tinted header band.
VARIANTS = [
    ("Described header band", "shadcn card anatomy + Blender panel header",
     dict(headerBand=True, showDesc=True, footerDivider=True, cardRadius=10, cardPad=14, radius=5)),
]


def tokens(theme_name, variant_index, font=""):
    c = THEMES[theme_name]
    dark = theme_name != "Light"
    lay = dict(BASE_LAYOUT, **VARIANTS[variant_index][2])

    panel = c["canvas"] if dark else c["dock"]
    card = c["surface"] if dark else c["elevated"]
    if lay["ghost"]:
        card = panel = c["surface"] if dark else c["background"]
    if lay["cardTint"]:
        card = _mix(card, c["highlight"], 0.07 if dark else 0.05)
    if lay["topHighlight"] and dark:
        panel = _mix(c["canvas"], (0, 0, 0), 0.35)
    field = c["background"] if dark else c["background"]
    if lay["fieldStyle"] == "outline":
        field = c["background"] if dark else c["elevated"]

    border = c["border"]
    card_border = _mix(card, c["text"], 0.10) if lay["topHighlight"] else (c["border"] if dark else c["border_subtle"])
    header = _mix(card, panel, 0.8) if dark else _mix(card, c["dock"], 0.7)

    accent = c["highlight"]
    on_accent = c["background"]
    t = dict(
        lay,
        font=font,
        panel=_hex(panel), card=_hex(card), field=_hex(field),
        rule=_hex(c["border"] if dark else c["border_subtle"]),
        border=_hex(border),
        borderHover=_hex(_mix(border, c["text_dim"], 0.5)),
        bottomStroke="",
        cardBorderColor=_hex(card_border),
        cardHeader=_hex(header),
        highlightLine=_hex((255, 255, 255), 0.07 if dark else 0.9),
        shadowAlpha=0.30 if dark else 0.045,
        text=_hex(c["text"]), label=_hex(c["text_dim"]),
        faint=_hex(_mix(c["text_dim"], c["text_muted"], 0.45)),
        placeholder=_hex(_mix(c["text_muted"], c["text_dim"], 0.3 if dark else 0.0)),
        accent=_hex(accent), accentText=_hex(accent), onAccent=_hex(on_accent), rail=_hex(accent),
        buttonBg=_hex(accent), buttonHover=_hex(c["highlight_hover"]), buttonText=_hex(on_accent),
        secondaryBg="transparent" if lay["fieldStyle"] in ("outline", "quiet") else _hex(field),
        track=_hex(c["border"]),
        handle=_hex(c["text"] if dark else c["elevated"]),
    )
    if lay["tonal"]:
        t.update(buttonBg=_hex(_mix(card, accent, 0.22)), buttonHover=_hex(_mix(card, accent, 0.32)),
                 buttonText=_hex(c["highlight_hover"] if dark else c["highlight_dim"]))
    return t
