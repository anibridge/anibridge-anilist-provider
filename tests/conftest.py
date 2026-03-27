"""Common pytest fixtures for the AniList provider test-suite."""

import logging
from collections.abc import Callable
from typing import cast

import pytest
from anibridge.utils.types import ProviderLogger

from anibridge.providers.list.anilist.client import AnilistClient
from anibridge.providers.list.anilist.list import AnilistListEntry, AnilistListProvider
from anibridge.providers.list.anilist.models import (
    Media,
    MediaCoverImage,
    MediaFormat,
    MediaList,
    MediaListStatus,
    MediaStatus,
    MediaTitle,
    ScoreFormat,
)
from tests.fakes import FakeAnilistClient


@pytest.fixture()
def media_factory() -> Callable[[int, str], Media]:
    """Return a factory that builds Media instances with sane defaults."""

    def _build(media_id: int, title: str) -> Media:
        entry = MediaList(
            id=media_id * 10,
            user_id=1,
            media_id=media_id,
            status=MediaListStatus.CURRENT,
            progress=3,
            repeat=0,
            score=75.0,
            started_at=None,
            completed_at=None,
        )
        return Media(
            id=media_id,
            title=MediaTitle(romaji=title, english=title.title()),
            format=MediaFormat.TV,
            status=MediaStatus.RELEASING,
            episodes=24,
            cover_image=MediaCoverImage(extra_large="xl.jpg", medium="m.jpg"),
            media_list_entry=entry,
        )

    return _build


@pytest.fixture()
def sample_media(media_factory: Callable[[int, str], Media]) -> Media:
    """Cowboy Bebop style sample media entry."""
    return media_factory(101, "cowboy bebop")


@pytest.fixture()
def second_media(media_factory: Callable[[int, str], Media]) -> Media:
    """Another sample media entry to use in batching tests."""
    media = media_factory(202, "your name")
    media.format = MediaFormat.MOVIE
    if not media.media_list_entry:
        raise ValueError("MediaListEntry must be present on media.")
    media.media_list_entry.status = MediaListStatus.PLANNING
    media.media_list_entry.progress = 0
    return media


@pytest.fixture()
def fake_client(sample_media: Media, second_media: Media) -> FakeAnilistClient:
    """Provide a fake AniList client seeded with deterministic media objects."""
    client = FakeAnilistClient(
        {sample_media.id: sample_media, second_media.id: second_media}
    )
    client.search_results = [sample_media, second_media]
    return client


@pytest.fixture()
def provider(fake_client: FakeAnilistClient) -> AnilistListProvider:
    """Return an AnilistListProvider wired to the fake client."""
    provider = AnilistListProvider(
        logger=cast(ProviderLogger, logging.getLogger("tests.provider")),
        config={"token": "fake-token", "profile_name": "pytest"},
    )
    provider._client = cast(AnilistClient, fake_client)
    provider._score_format = ScoreFormat.POINT_10_DECIMAL
    return provider


@pytest.fixture()
def entry_factory(provider: AnilistListProvider) -> Callable[[Media], AnilistListEntry]:
    """Return a helper that wraps media objects in AnilistListEntry instances."""

    def _build(media: Media) -> AnilistListEntry:
        assert media.media_list_entry is not None
        return AnilistListEntry(provider, media=media, entry=media.media_list_entry)

    return _build


@pytest.fixture(autouse=True)
def disable_rate_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove runtime wrappers so tests stay fast and loop-safe."""
    wrapped = getattr(AnilistClient._make_request, "__wrapped__", None)
    if wrapped is not None:
        monkeypatch.setattr(AnilistClient, "_make_request", wrapped)
    search_wrapped = getattr(AnilistClient._search_anime, "__wrapped__", None)
    if search_wrapped is not None:
        monkeypatch.setattr(AnilistClient, "_search_anime", search_wrapped)
