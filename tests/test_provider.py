"""Tests for the AniList provider contract."""

from datetime import UTC, date, datetime
from logging import getLogger
from typing import Any, cast

import pytest
from anibridge.provider.base import (
    Artwork,
    FacetName,
    Node,
    NodeKind,
    NodeQuery,
    NodeSpec,
    Record,
    RecordField,
    RecordQuery,
    Ref,
    ResourceKind,
    SupportsReads,
    UpsertRecord,
)

from anibridge.providers.anilist.client import AnilistClient
from anibridge.providers.anilist.models import FuzzyDate, MediaListWithMedia
from anibridge.providers.anilist.provider import AnilistProvider


@pytest.fixture()
def provider(fake_client: Any) -> AnilistProvider:
    """Return an AniList provider wired to the fake client."""
    provider = AnilistProvider(
        logger=getLogger("tests.provider"),
        config={"token": "fake-token"},
    )
    provider._client = cast(AnilistClient, fake_client)
    return provider


def test_capabilities_advertise_node_reads(provider: AnilistProvider) -> None:
    """AniList should expose targeted node metadata for web timeline enrichment."""
    capabilities = provider.capabilities()

    assert isinstance(provider, SupportsReads)
    assert FacetName.ARTWORK in capabilities.facets
    node_spec = next(
        spec for spec in capabilities.specs if spec.resource is ResourceKind.NODE
    )
    node_spec = cast(NodeSpec, node_spec)
    assert node_spec.kind.native == "anime"
    assert node_spec.kind.semantic == NodeKind.SERIES


@pytest.mark.asyncio
async def test_fetch_nodes_returns_anilist_metadata(provider: AnilistProvider) -> None:
    """Node reads should resolve titles, links, labels, and requested artwork."""
    page = await provider.fetch(
        NodeQuery(
            refs=(Ref.anchor("bad-ref"),),
        )
    )

    assert page.items == ()

    page = await provider.fetch(
        NodeQuery(
            refs=(Ref.anchor("101"),),
            facets=frozenset({FacetName.ARTWORK}),
        )
    )

    assert len(page.items) == 1
    node = cast(Node, page.items[0])
    assert node.ref == Ref.anchor("101")
    assert node.kind == "anime"
    assert node.title == "cowboy bebop"
    assert node.url == "https://anilist.co/anime/101"
    assert node.labels == ("Tv", "Releasing")
    artwork = node.facets[FacetName.ARTWORK]
    assert isinstance(artwork, Artwork)
    assert artwork.poster == "m.jpg"


@pytest.mark.asyncio
async def test_fetch_records_fetches_uncached_media(provider: AnilistProvider) -> None:
    """Record reads should fall back to targeted media fetches on cache misses."""
    page = await provider.fetch(RecordQuery(refs=(Ref.anchor("101"),)))

    assert len(page.items) == 1
    record = cast(Record, page.items[0])
    assert record.ref == Ref.anchor("101")
    assert record.surface == "media_list"


@pytest.mark.asyncio
async def test_upsert_record_returns_write_ref(
    provider: AnilistProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AniList write results should preserve the requested contract ref."""

    async def save_entry(_variables: dict[str, Any]) -> MediaListWithMedia:
        return MediaListWithMedia(
            id=2020,
            user_id=1,
            media_id=202,
            progress=4,
            updated_at=123,
        )

    monkeypatch.setattr(provider, "_save_media_list_entry", save_entry)
    write = UpsertRecord(
        ref=Ref.anchor("202"),
        surface="media_list",
        set={RecordField.PROGRESS: 4},
    )

    result = await provider._upsert_record(write)

    assert result.ok is True
    assert result.ref == write.ref


def test_record_from_media_preserves_anilist_date_precision(
    provider: AnilistProvider, sample_media: Any
) -> None:
    """AniList fuzzy dates should remain dates in normalized records."""
    assert sample_media.media_list_entry is not None
    sample_media.media_list_entry.started_at = FuzzyDate(
        year=2026,
        month=1,
        day=2,
    )
    sample_media.media_list_entry.completed_at = FuzzyDate(
        year=2026,
        month=1,
        day=3,
    )

    record = provider._record_from_media(sample_media, frozenset())

    assert record.values[RecordField.STARTED_AT] == date(2026, 1, 2)
    assert record.values[RecordField.FINISHED_AT] == date(2026, 1, 3)


def test_to_anilist_value_accepts_dates_for_date_precision(
    provider: AnilistProvider,
) -> None:
    """AniList writes should turn dates into FuzzyDateInput values."""
    assert provider._to_anilist_value(
        RecordField.STARTED_AT,
        date(2026, 1, 2),
    ) == {"year": 2026, "month": 1, "day": 2}
    assert provider._to_anilist_value(
        RecordField.FINISHED_AT,
        datetime(2026, 1, 3, 12, 30, tzinfo=UTC),
    ) == {"year": 2026, "month": 1, "day": 3}

    with pytest.raises(ValueError, match="timezone-aware"):
        provider._to_anilist_value(
            RecordField.STARTED_AT,
            datetime(2026, 1, 2),
        )
