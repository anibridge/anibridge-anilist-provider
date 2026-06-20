"""Tests for the AniList provider contract."""

from datetime import UTC, date, datetime
from logging import getLogger
from typing import Any, cast

import pytest
from anibridge.provider.base import (
    Artwork,
    FacetName,
    NodeKind,
    NodeQuery,
    RecordField,
    Ref,
    SupportsNodeReads,
)

from anibridge.providers.anilist.client import AnilistClient
from anibridge.providers.anilist.models import FuzzyDate
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

    assert isinstance(provider, SupportsNodeReads)
    assert FacetName.ARTWORK in capabilities.facets
    assert capabilities.nodes[0].kind.native == "anime"
    assert capabilities.nodes[0].kind.semantic == NodeKind.SERIES


@pytest.mark.asyncio
async def test_fetch_nodes_returns_anilist_metadata(provider: AnilistProvider) -> None:
    """Node reads should resolve titles, links, labels, and requested artwork."""
    page = await provider.fetch_nodes(
        NodeQuery(
            refs=(Ref.anchor("bad-ref"),),
        )
    )

    assert page.items == ()

    page = await provider.fetch_nodes(
        NodeQuery(
            refs=(Ref.anchor("101"),),
            facets=frozenset({FacetName.ARTWORK}),
        )
    )

    assert len(page.items) == 1
    node = page.items[0]
    assert node.ref == Ref.anchor("101")
    assert node.kind == "anime"
    assert node.title == "cowboy bebop"
    assert node.url == "https://anilist.co/anime/101"
    assert node.labels == ("Tv", "Releasing")
    artwork = node.facets[FacetName.ARTWORK]
    assert isinstance(artwork, Artwork)
    assert artwork.poster == "m.jpg"


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
