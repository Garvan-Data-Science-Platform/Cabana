"""What's New notes: when they show and which notes are picked."""
from cabana import whats_new


def test_shown_only_after_an_upgrade_to_a_version_with_notes(monkeypatch):
    monkeypatch.setattr(whats_new, "NOTES", {"1.2.0": ["New thing."]})
    assert whats_new.should_show("1.1.5", "1.2.0")
    assert not whats_new.should_show("", "1.2.0")            # fresh install: the tour is offered instead
    assert not whats_new.should_show("1.2.0", "1.2.0")       # already seen
    assert not whats_new.should_show("1.2.0", "1.1.5")       # downgrade
    assert not whats_new.should_show("1.2.0", "1.3.0")       # no notes for this version


def test_latest_is_the_newest_release_with_notes(monkeypatch):
    monkeypatch.setattr(whats_new, "NOTES", {"0.3.9": ["a"], "0.3.18": ["b"], "0.3.10": ["c"]})
    assert whats_new.latest() == ("0.3.18", ["b"])
    assert whats_new.notes_for("0.3.9") == ["a"] and whats_new.notes_for("9.9.9") is None


def test_every_entry_is_brief():
    for version, notes in whats_new.NOTES.items():
        assert 1 <= len(notes) <= 4, version
        assert all(notes) and all(len(n) < 160 for n in notes), version


def test_a_skipped_release_still_shows_its_notes(monkeypatch):
    monkeypatch.setattr(whats_new, "NOTES", {"1.2.0": ["Big change."], "1.2.1": ["Small fix."]})
    assert whats_new.notes_since("1.1.0", "1.2.1") == [("1.2.1", ["Small fix."]), ("1.2.0", ["Big change."])]
    assert whats_new.notes_since("1.2.0", "1.2.1") == [("1.2.1", ["Small fix."])]
    assert whats_new.should_show("1.1.0", "1.2.2")             # no notes for 1.2.2, but for the skipped ones
