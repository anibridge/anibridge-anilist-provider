"""Shared test doubles for the AniList provider tests."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC

from anibridge.providers.list.anilist.models import (
    Media,
    MediaList,
    MediaListOptions,
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
