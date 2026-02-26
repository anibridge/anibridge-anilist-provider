"""Common pytest fixtures for the AniList provider test-suite."""

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC
from typing import cast

import pytest

from anibridge_anilist_provider.client import AnilistClient
from anibridge_anilist_provider.list import AnilistListEntry, AnilistListProvider
from anibridge_anilist_provider.models import (
    Media,
    MediaCoverImage,
    MediaFormat,
    MediaList,
    MediaListOptions,
    MediaListStatus,
    MediaStatus,
    MediaTitle,
    ScoreFormat,
    User,
)


@dataclass
class FakeAnilistClient:
    """Minimal asynchronous client stand-in used by the tests."""

    medias: dict[int, Media] = field(default_factory=dict)
    search_results: list[Media] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize default user and state."""
        self.user = User.model_construct(
            id=99,
            name="Test User",
            media_list_options=MediaListOptions(
                score_format=ScoreFormat.POINT_10_DECIMAL
            ),
        )
        self.offline_anilist_entries: dict[int, Media] = {}
        self.update_payloads: list[MediaList] = []
        self.batch_update_payloads: list[list[MediaList]] = []
        self.deleted_entries: list[tuple[int, int]] = []
        self.user_timezone = UTC

        if not self.search_results:
            self.search_results = list(self.medias.values())

    async def get_anime(self, media_id: int) -> Media:
        """Retrieve a Media by its AniList ID."""
        return self.medias[media_id]

    async def batch_get_anime(self, ids: list[int]) -> list[Media]:
        """Retrieve multiple Media by their AniList IDs."""
        return [self.medias[id] for id in ids if id in self.medias]

    async def search_anime(
        self, query: str, is_movie: bool | None, limit: int
    ) -> AsyncIterator[Media]:
        """Yield Media matching the query up to the specified limit."""
        lowered = query.lower()
        count = 0
        for media in self.search_results:
            if not media.title:
                continue
            title = (media.title.romaji or media.title.english or "").lower()
            if lowered in title and count < limit:
                count += 1
                yield media

    async def update_anime_entry(self, payload: MediaList) -> None:
        """Record an update payload for later inspection."""
        self.update_payloads.append(payload)

    async def batch_update_anime_entries(self, payloads: list[MediaList]) -> set[int]:
        """Record a batch update payload for later inspection."""
        self.batch_update_payloads.append(payloads)
        return {payload.media_id for payload in payloads}

    async def delete_anime_entry(self, entry_id: int, media_id: int) -> None:
        """Record delete requests for verification in tests."""
        self.deleted_entries.append((entry_id, media_id))

    async def initialize(self) -> None:
        """No-op initialization method."""
        return None

    async def close(self) -> None:
        """No-op close method."""
        return None


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
        logger=logging.getLogger("tests.provider"),
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
