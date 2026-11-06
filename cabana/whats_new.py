"""Short release notes shown once after an upgrade; Help > What's New shows
them at any time.

Add an entry for every release: two to four user-facing points in the style
of the changelog (docs/source/changelog.md), which holds the full list.
"""
from .updates import is_newer

NOTES = {
    "0.3.18": [
        "A cleaner side panel: settings are grouped in cards that fold away, with each "
        "step's main action at the bottom of its card.",
        "Help > Take the Tour walks through the analysis, pointing at each step.",
        "Only the Light and Dark themes remain.",
        "Fixed: choosing the Batch Run image folder before the parameter file failed.",
    ],
}


def notes_for(version):
    """The notes of ``version``, or None when it has none."""
    return NOTES.get(version)


def latest():
    """(version, notes) of the newest release that has notes."""
    version = max(NOTES, key=lambda v: tuple(int(p) for p in v.split(".") if p.isdigit()))
    return version, NOTES[version]


def should_show(last_seen, current):
    """True after an upgrade to a version that has notes (not on a fresh install)."""
    return bool(last_seen) and is_newer(current, last_seen) and notes_for(current) is not None
