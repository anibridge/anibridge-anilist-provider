"""Model-focused unit tests for AniList helpers."""

from datetime import UTC, date, datetime

import pytest

from anibridge_anilist_provider.models import (
    FuzzyDate,
    MediaListStatus,
    MediaListWithMedia,
    MediaTitle,
)


def _fuzzy(**kwargs) -> FuzzyDate:
    """Helper for constructing FuzzyDate instances without validation noise."""
    return FuzzyDate.model_construct(**kwargs)


def test_media_list_status_priority():
    """Statuses should compare according to the configured priority map."""
    assert MediaListStatus.COMPLETED > MediaListStatus.CURRENT
    assert MediaListStatus.PAUSED <= MediaListStatus.CURRENT
    assert MediaListStatus.PLANNING < MediaListStatus.DROPPED


@pytest.mark.parametrize(
    "value",
    [date(2023, 5, 20), datetime(2023, 5, 20, tzinfo=UTC)],
)
def test_fuzzy_date_conversion_roundtrip(value):
    """FuzzyDate.from_date should preserve year/month/day information."""
    fuzzy = FuzzyDate.from_date(value)
    assert bool(fuzzy)
    assert fuzzy.year == 2023
    assert fuzzy.month == 5
    assert fuzzy.day == 20
    assert fuzzy.to_datetime() == datetime(2023, 5, 20)


def test_fuzzy_date_comparison_handles_missing_fields():
    """Ordering should treat incomplete dates as lower precision values."""
    incomplete = _fuzzy(year=2023)
    precise = _fuzzy(year=2023, month=6, day=1)
    assert incomplete < precise
    assert not (precise < incomplete)


def test_fuzzy_date_boolean_false_without_year():
    """Fuzzy dates without a year should evaluate to False in boolean context."""
    fuzzy = _fuzzy(month=5, day=12)
    assert not fuzzy


def test_fuzzy_date_to_datetime_defaults_month_and_day():
    """to_datetime should default missing components to the first of the month."""
    fuzzy = _fuzzy(year=2024)
    assert fuzzy.to_datetime() == datetime(2024, 1, 1)


def test_fuzzy_date_from_date_handles_none():
    """from_date should propagate None inputs without raising."""
    assert FuzzyDate.from_date(None) is None


def test_media_list_status_comparisons_with_other_types():
    """MediaListStatus comparison methods should return NotImplemented for strangers."""
    status = MediaListStatus.CURRENT
    assert status.__eq__("current") is NotImplemented
    assert status.__ne__(123) is NotImplemented
    assert status.__lt__(object()) is NotImplemented
    assert status.__le__(object()) is NotImplemented
    assert status.__gt__(object()) is NotImplemented
    assert status.__ge__(object()) is NotImplemented


def test_base_model_unset_and_dump_helpers():
    """AniListBaseModel helpers should reset fields and honor aliasing."""
    title = MediaTitle(romaji="romaji", english="english")
    title.unset_fields(["english"])
    assert title.english is None
    dumped = title.model_dump()
    assert "romaji" in dumped
    assert "english" in dumped and dumped["english"] is None
    json_payload = title.model_dump_json()
    assert "romaji" in json_payload


def test_model_dump_graphql_handles_nested_and_cycles():
    """model_dump_graphql should emit nested structures and guard recursion."""
    fields = MediaListWithMedia.model_dump_graphql()
    assert "media {" in fields

    MediaTitle.model_dump_graphql.cache_clear()
    MediaTitle._processed_models.add("MediaTitle")
    try:
        assert MediaTitle.model_dump_graphql() == ""
    finally:
        MediaTitle.model_dump_graphql.cache_clear()
        MediaTitle._processed_models.discard("MediaTitle")


def test_base_model_repr_and_hash_consistency():
    """__repr__ should drive __hash__ for deterministic values."""
    title = MediaTitle(romaji="repr test")
    repr_value = repr(title)
    assert repr_value.startswith("<")
    assert hash(title) == hash(repr_value)


def test_media_title_titles_and_string_fallbacks():
    """MediaTitle should return all configured titles and string fallbacks."""
    title = MediaTitle(romaji="romaji", native="native")
    titles = title.titles()
    assert titles[0] == "romaji"
    assert str(title) == "romaji"


def test_fuzzy_date_to_datetime_without_year():
    """FuzzyDate with no year should return None when converted to datetime."""
    assert _fuzzy(month=5).to_datetime() is None


def test_fuzzy_date_equality_with_other_type():
    """Equality checks should return False for non-FuzzyDate objects."""
    assert _fuzzy(year=2020) != "2020"


def test_fuzzy_date_comparison_missing_year_defaults():
    """Comparisons with missing years should short-circuit to True."""
    lhs = _fuzzy(month=1)
    rhs = _fuzzy(year=2024)
    assert lhs < rhs
    assert lhs <= rhs
    assert lhs > rhs
    assert lhs >= rhs


def test_fuzzy_date_comparison_notimplemented_for_other_types():
    """Ordering comparisons should return NotImplemented for other types."""
    date = _fuzzy(year=2023)
    marker = object()
    assert FuzzyDate.__lt__(date, marker) is NotImplemented
    assert FuzzyDate.__le__(date, marker) is NotImplemented
    assert FuzzyDate.__gt__(date, marker) is NotImplemented
    assert FuzzyDate.__ge__(date, marker) is NotImplemented


def test_fuzzy_date_repr_handles_unknown_components():
    """__repr__ should pad unknown values with placeholders."""
    assert repr(_fuzzy()) == "????-??-??"
