"""Tests that focus on the AniList list provider behavior."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest
from anibridge.list import ListMediaType, ListStatus

from anibridge_anilist_provider.client import AniListClient
from anibridge_anilist_provider.list import (
    AniListListEntry,
    AniListListMedia,
    AniListListProvider,
)
from anibridge_anilist_provider.models import (
    FuzzyDate,
    Media,
    MediaCoverImage,
    MediaFormat,
    MediaListOptions,
    MediaListStatus,
    ScoreFormat,
)
from anibridge_anilist_provider.models import (
    User as AniListAPIUser,
)

if TYPE_CHECKING:
    from tests.conftest import FakeAniListClient


@pytest.mark.asyncio
async def test_build_media_payload_maps_entry_state(
    provider: AniListListProvider,
    sample_media: Media,
    entry_factory: Callable[[Media], AniListListEntry],
):
    """Ensure _build_media_payload honors the entry transformations."""
    entry = entry_factory(sample_media)
    entry.status = ListStatus.COMPLETED
    entry.progress = 24
    entry.repeats = 2
    entry.review = "Great show"
    entry.user_rating = 80
    start = datetime(2024, 1, 1, tzinfo=UTC)
    finish = datetime(2024, 2, 1, tzinfo=UTC)
    entry.started_at = start
    entry.finished_at = finish

    payload = await provider._build_media_payload(sample_media.id, entry)

    assert payload.status == MediaListStatus.COMPLETED
    assert payload.progress == 24
    assert payload.repeat == 2
    assert payload.notes == "Great show"
    assert payload.score == pytest.approx(8.0)
    assert payload.started_at == FuzzyDate.from_date(start)
    assert payload.completed_at == FuzzyDate.from_date(finish)


@pytest.mark.asyncio
async def test_update_entries_batch_calls_client_once(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
    sample_media: Media,
    second_media: Media,
    entry_factory: Callable[[Media], AniListListEntry],
):
    """Batch updates should forward payloads to the underlying client."""
    entry_one = entry_factory(sample_media)
    entry_two = entry_factory(second_media)

    await provider.update_entries_batch([entry_one, entry_two])

    assert len(fake_client.batch_update_payloads) == 1
    payloads = fake_client.batch_update_payloads[0]
    assert {payload.media_id for payload in payloads} == {
        sample_media.id,
        second_media.id,
    }


@pytest.mark.asyncio
async def test_get_entries_batch_preserves_order(
    provider: AniListListProvider,
    second_media: Media,
):
    """Missing items should map to None while preserving input ordering."""
    results = await provider.get_entries_batch([str(second_media.id), "404"])
    assert results[0] is not None
    assert isinstance(results[0], AniListListEntry)
    assert results[0].media().key == str(second_media.id)
    assert results[1] is None


@pytest.mark.asyncio
async def test_search_wraps_media_into_entries(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
    sample_media: Media,
):
    """Search should adapt AniList media results into AniBridge entries."""
    fake_client.search_results = [sample_media]

    results = await provider.search("cowboy")

    assert len(results) == 1
    entry = results[0]
    assert isinstance(entry, AniListListEntry)
    assert entry.media().key == str(sample_media.id)
    assert entry.status == ListStatus.CURRENT


def test_entry_progress_validation(
    entry_factory: Callable[[Media], AniListListEntry], sample_media: Media
):
    """Progress setter should reject negative numbers."""
    entry = entry_factory(sample_media)
    with pytest.raises(ValueError):
        entry.progress = -1


def test_entry_user_rating_respects_score_format(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
    sample_media: Media,
    entry_factory: Callable[[Media], AniListListEntry],
):
    """User rating should convert between AniList formats and 0-100 scale."""
    if fake_client.user.media_list_options is None:
        raise ValueError("User media list options must be set for this test.")
    fake_client.user.media_list_options.score_format = ScoreFormat.POINT_5
    provider._score_format = ScoreFormat.POINT_5
    entry = entry_factory(sample_media)

    entry.user_rating = 80
    assert entry._entry.score == 4  # POINT_5 scale

    entry._entry.score = 4
    assert entry.user_rating == 80


class StubAniListClient:
    """Lightweight stub that mimics AniListClient behaviors for provider tests."""

    def __init__(self) -> None:
        """Initialize stub state containers."""
        self.initialize_called = False
        self.backup_value = "backup-data"
        self.restored_payload: str | None = None

    async def initialize(self) -> None:
        """Record that initialization was invoked."""
        self.initialize_called = True

    async def get_user(self) -> AniListAPIUser:
        """Return a deterministic AniList API user payload."""
        return AniListAPIUser(
            id=1,
            name="Remote User",
            media_list_options=MediaListOptions(score_format=ScoreFormat.POINT_5),
        )

    async def backup_anilist(self) -> str:
        """Return the canned backup payload."""
        return self.backup_value

    async def restore_anilist(self, payload: str) -> None:
        """Record the last restore payload for assertions."""
        self.restored_payload = payload

    async def close(self) -> None:
        """No-op close hook for interface parity."""
        return None


@pytest.mark.parametrize(
    ("score_format", "rating", "expected_internal"),
    [
        (ScoreFormat.POINT_100, 55, pytest.approx(55.0)),
        (ScoreFormat.POINT_10_DECIMAL, 83, pytest.approx(8.3)),
        (ScoreFormat.POINT_10, 70, 7),
        (ScoreFormat.POINT_5, 80, 4),
        (ScoreFormat.POINT_3, 66, pytest.approx(1.98)),
    ],
)
def test_entry_user_rating_setter_handles_all_formats(
    provider: AniListListProvider,
    sample_media: Media,
    entry_factory: Callable[[Media], AniListListEntry],
    score_format: ScoreFormat,
    rating: int,
    expected_internal: float,
):
    """Ensure setter writes AniList-native values for each score format."""
    entry = entry_factory(sample_media)
    provider._score_format = score_format
    entry._entry.score = None

    entry.user_rating = rating

    assert entry._entry.score == expected_internal


@pytest.mark.parametrize(
    ("score_format", "stored", "expected_rating"),
    [
        (ScoreFormat.POINT_100, 72.0, 72),
        (ScoreFormat.POINT_10_DECIMAL, 8.6, 86),
        (ScoreFormat.POINT_10, 7, 70),
        (ScoreFormat.POINT_5, 4, 80),
        (ScoreFormat.POINT_3, 2.5, 83),
    ],
)
def test_entry_user_rating_getter_honors_client_format(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
    sample_media: Media,
    entry_factory: Callable[[Media], AniListListEntry],
    score_format: ScoreFormat,
    stored: float,
    expected_rating: int,
):
    """Confirm getter normalizes AniList scores back to 0-100 scale."""
    entry = entry_factory(sample_media)
    options = fake_client.user.media_list_options
    if options is None:
        options = MediaListOptions()
        fake_client.user.media_list_options = options
    options.score_format = score_format
    entry._entry.score = stored

    assert entry.user_rating == expected_rating


def test_entry_user_rating_rejects_out_of_range(
    provider: AniListListProvider,
    sample_media: Media,
    entry_factory: Callable[[Media], AniListListEntry],
):
    """Ratings above the allowed range should be rejected."""
    entry = entry_factory(sample_media)

    with pytest.raises(ValueError):
        entry.user_rating = 101


@pytest.mark.parametrize(
    ("media_status", "expected"),
    [
        (MediaListStatus.DROPPED, ListStatus.DROPPED),
        (MediaListStatus.PAUSED, ListStatus.PAUSED),
        (MediaListStatus.PLANNING, ListStatus.PLANNING),
        (MediaListStatus.REPEATING, ListStatus.REPEATING),
    ],
)
def test_entry_status_getter_handles_media_statuses(
    entry_factory: Callable[[Media], AniListListEntry],
    sample_media: Media,
    media_status: MediaListStatus,
    expected: ListStatus,
):
    """Status getter should translate every MediaListStatus variant."""
    entry = entry_factory(sample_media)
    entry._entry.status = media_status

    assert entry.status == expected


def test_entry_status_setter_rejects_unknown_value(
    entry_factory: Callable[[Media], AniListListEntry], sample_media: Media
):
    """Setting an unsupported status should raise a ValueError."""
    entry = entry_factory(sample_media)

    with pytest.raises(ValueError):
        entry.status = cast(ListStatus, object())


def test_entry_progress_allows_none(entry_factory, sample_media):
    """Progress setter should allow clearing the stored value."""
    entry = entry_factory(sample_media)
    entry._entry.progress = 5

    entry.progress = None

    assert entry._entry.progress is None


def test_entry_repeats_validation(
    entry_factory: Callable[[Media], AniListListEntry], sample_media: Media
):
    """Repeat setter should allow None but reject negatives."""
    entry = entry_factory(sample_media)
    entry._entry.repeat = 1

    entry.repeats = None
    assert entry._entry.repeat is None

    with pytest.raises(ValueError):
        entry.repeats = -2


def test_entry_user_rating_returns_none_when_unset(
    entry_factory: Callable[[Media], AniListListEntry], sample_media: Media
):
    """user_rating should return None when AniList lacks a score."""
    entry = entry_factory(sample_media)
    entry._entry.score = None

    assert entry.user_rating is None


def test_entry_user_rating_getter_defaults_without_format(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
    sample_media: Media,
    entry_factory: Callable[[Media], AniListListEntry],
):
    """If AniList omits the score format the getter should fall back to ints."""
    entry = entry_factory(sample_media)
    options = fake_client.user.media_list_options
    if options is None:
        options = MediaListOptions()
        fake_client.user.media_list_options = options
    options.score_format = None
    entry._entry.score = 47.6

    assert entry.user_rating == 47


def test_entry_user_rating_setter_handles_none_and_unknown_format(
    provider: AniListListProvider,
    sample_media: Media,
    entry_factory: Callable[[Media], AniListListEntry],
):
    """Setter should clear scores and fall back to float values when needed."""
    entry = entry_factory(sample_media)
    entry.user_rating = None
    assert entry._entry.score is None

    provider._score_format = cast(ScoreFormat, object())
    entry.user_rating = 25

    assert entry._entry.score == pytest.approx(25.0)


def test_entry_total_units_prefers_media_total(
    entry_factory: Callable[[Media], AniListListEntry],
    media_factory: Callable[[int, str], Media],
):
    """Entry.total_units should return the media's explicit episode count."""
    media = media_factory(111, "seasonal")
    media.episodes = 13

    assert entry_factory(media).total_units == 13


def test_entry_total_units_returns_none_for_tv_without_data(
    entry_factory: Callable[[Media], AniListListEntry],
    media_factory: Callable[[int, str], Media],
):
    """TV entries without total units and unknown episodes should return None."""
    media = media_factory(222, "mystery show")
    media.episodes = None

    assert entry_factory(media).total_units is None


def test_entry_media_and_provider_accessors(
    provider: AniListListProvider,
    entry_factory: Callable[[Media], AniListListEntry],
    sample_media: Media,
):
    """Entry helpers should expose the underlying media and provider."""
    entry = entry_factory(sample_media)

    assert entry.media().provider() is provider
    assert entry.provider() is provider


@pytest.mark.parametrize("status", list(ListStatus))
def test_entry_status_roundtrip(
    entry_factory: Callable[[Media], AniListListEntry],
    sample_media: Media,
    status: ListStatus,
):
    """Status setter/getter pair should round-trip all supported statuses."""
    entry = entry_factory(sample_media)

    entry.status = status

    assert entry.status == status


def test_provider_requires_token() -> None:
    """Provider construction without a token should raise a ValueError."""
    with pytest.raises(ValueError):
        AniListListProvider(config={})


@pytest.mark.asyncio
async def test_provider_initialize_sets_user_and_score_format() -> None:
    """Initialize should cache the provider user metadata and score format."""
    provider = AniListListProvider(config={"token": "abc"})
    stub = StubAniListClient()
    provider._client = cast(AniListClient, stub)

    await provider.initialize()

    user = provider.user()
    assert user is not None
    assert user.title == "Remote User"
    assert provider._score_format == ScoreFormat.POINT_5
    assert stub.initialize_called


@pytest.mark.asyncio
async def test_backup_and_restore_list_delegate_to_client() -> None:
    """backup_list and restore_list should proxy to the underlying client."""
    provider = AniListListProvider(config={"token": "abc"})
    stub = StubAniListClient()
    provider._client = cast(AniListClient, stub)

    backup_payload = await provider.backup_list()
    await provider.restore_list("payload")

    assert backup_payload == "backup-data"
    assert stub.restored_payload == "payload"


@pytest.mark.asyncio
async def test_provider_close_forwards_to_client() -> None:
    """Close should call through to the AniList client."""
    provider = AniListListProvider(config={"token": "abc"})
    stub = StubAniListClient()
    provider._client = cast(AniListClient, stub)

    await provider.close()
    # close() doesn't expose state, but ensure stub is still usable
    assert stub.restored_payload is None


@pytest.mark.asyncio
async def test_get_entry_returns_none_without_media_list(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
    media_factory: Callable[[int, str], Media],
):
    """get_entry should return None when AniList lacks a list entry."""
    orphan = media_factory(909, "no list")
    orphan.media_list_entry = None
    fake_client.medias[orphan.id] = orphan

    assert await provider.get_entry(str(orphan.id)) is None


@pytest.mark.asyncio
async def test_get_entries_batch_handles_empty_input(provider: AniListListProvider):
    """get_entries_batch should preserve list length even when empty."""
    assert await provider.get_entries_batch([]) == []


@pytest.mark.asyncio
async def test_update_entries_batch_handles_empty_sequence(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
):
    """update_entries_batch should bypass the client when no entries exist."""
    assert await provider.update_entries_batch([]) == []
    assert fake_client.batch_update_payloads == []


@pytest.mark.asyncio
async def test_build_media_payload_uses_base_status_when_entry_missing(
    provider: AniListListProvider,
    sample_media: Media,
    entry_factory: Callable[[Media], AniListListEntry],
):
    """_build_media_payload should fall back to AniList's current status."""
    entry = entry_factory(sample_media)
    entry.status = None
    assert sample_media.media_list_entry is not None

    payload = await provider._build_media_payload(sample_media.id, entry)

    assert payload.status == sample_media.media_list_entry.status


@pytest.mark.asyncio
async def test_build_entry_creates_placeholder_when_missing(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
    media_factory: Callable[[int, str], Media],
):
    """Provider.build_entry should fabricate list data when AniList lacks it."""
    orphan_media = media_factory(303, "march comes in like a lion")
    orphan_media.media_list_entry = None
    fake_client.medias[orphan_media.id] = orphan_media

    entry = await provider.build_entry(str(orphan_media.id))

    assert entry.media().key == str(orphan_media.id)
    assert entry.progress == 0
    assert entry.status is None
    assert entry._entry.id == 0


def test_media_total_units_defaults_for_movies(
    entry_factory: Callable[[Media], AniListListEntry],
    media_factory: Callable[[int, str], Media],
):
    """Movie entries without an explicit episode count should default to 1."""
    media = media_factory(404, "the movie")
    assert media.media_list_entry is not None
    media.format = MediaFormat.MOVIE
    media.episodes = None
    entry = entry_factory(media)

    assert entry.media().total_units == 1


def test_entry_total_units_falls_back_for_non_tv_media(
    entry_factory: Callable[[Media], AniListListEntry],
    media_factory: Callable[[int, str], Media],
):
    """OVA-like formats should default to a single unit when missing episodes."""
    media = media_factory(505, "ova special")
    assert media.media_list_entry is not None
    media.format = MediaFormat.OVA
    media.episodes = None
    entry = entry_factory(media)

    assert entry.total_units == 1


def test_list_media_media_type_and_provider(
    provider: AniListListProvider,
    media_factory: Callable[[int, str], Media],
):
    """AniListListMedia should expose both the provider and media_type."""
    tv_media = media_factory(606, "tv show")
    tv = AniListListMedia(provider, tv_media)
    assert tv.media_type == ListMediaType.TV
    assert tv.provider() is provider

    movie_media = media_factory(707, "movie")
    movie_media.format = MediaFormat.MOVIE
    movie = AniListListMedia(provider, movie_media)
    assert movie.media_type == ListMediaType.MOVIE


def test_list_media_total_units_prefers_episode_count(
    provider: AniListListProvider,
    media_factory: Callable[[int, str], Media],
):
    """When AniList exposes an episode count it should be used directly."""
    media = media_factory(808, "long show")
    media.episodes = 52

    assert AniListListMedia(provider, media).total_units == 52


def test_list_media_poster_image_handles_missing_cover(
    provider: AniListListProvider,
    media_factory: Callable[[int, str], Media],
):
    """If the cover image data is missing the poster should be None."""
    media = media_factory(909, "coverless")
    media.cover_image = None

    assert AniListListMedia(provider, media).poster_image is None


def test_media_poster_image_falls_back_to_color(
    entry_factory: Callable[[Media], AniListListEntry],
    media_factory: Callable[[int, str], Media],
):
    """Poster image selection should walk the cover image fallbacks."""
    media = media_factory(606, "coverless show")
    assert media.media_list_entry is not None
    media.cover_image = MediaCoverImage(color="#123456")
    entry = entry_factory(media)

    assert entry.media().poster_image == "#123456"


@pytest.mark.asyncio
async def test_clear_cache_empties_fake_storage(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
    sample_media: Media,
):
    """clear_cache should drop any cached media entries on the client."""
    fake_client.offline_anilist_entries = {sample_media.id: sample_media}

    await provider.clear_cache()

    assert fake_client.offline_anilist_entries == {}


@pytest.mark.asyncio
async def test_delete_entry_triggers_client_delete(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
    sample_media: Media,
):
    """Deleting an entry should forward the AniList identifiers to the client."""
    assert sample_media.media_list_entry is not None

    await provider.delete_entry(str(sample_media.id))

    assert fake_client.deleted_entries == [
        (sample_media.media_list_entry.id, sample_media.id)
    ]


@pytest.mark.asyncio
async def test_delete_entry_noop_without_media_list(
    provider: AniListListProvider,
    fake_client: FakeAniListClient,
    media_factory: Callable[[int, str], Media],
):
    """Deleting when AniList lacks an entry should not hit the client."""
    orphan_media = media_factory(808, "orphan")
    orphan_media.media_list_entry = None
    fake_client.medias[orphan_media.id] = orphan_media

    await provider.delete_entry(str(orphan_media.id))

    assert fake_client.deleted_entries == []
