"""Unit tests focusing on the AnilistClient helper behaviors."""

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any, cast

import aiohttp
import pytest
from anibridge.utils.types import ProviderLogger

from anibridge.providers.list.anilist.client import AnilistClient
from anibridge.providers.list.anilist.models import (
    Media,
    MediaFormat,
    MediaList,
    MediaListCollection,
    MediaListCollectionWithMedia,
    MediaListGroup,
    MediaListGroupWithMedia,
    MediaListStatus,
    MediaListWithMedia,
    MediaStatus,
    MediaTitle,
    MediaWithoutList,
    User,
    UserOptions,
)


@pytest.fixture()
def client() -> AnilistClient:
    """Return a fresh AnilistClient instance backed by the stubbed token."""
    return AnilistClient(
        anilist_token="token",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
    )


def test_default_rate_limiter_is_shared_across_clients() -> None:
    """Clients without a custom limit should reuse one global limiter."""
    first = AnilistClient(
        anilist_token="token",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
    )
    second = AnilistClient(
        anilist_token="token",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
    )

    assert first.rate_limit is None
    assert second.rate_limit is None
    assert first._request_limiter is second._request_limiter


def test_custom_rate_limiter_is_local_per_client() -> None:
    """Custom limits should create per-client limiters and convert to req/sec."""
    first = AnilistClient(
        anilist_token="token",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
        rate_limit=120,
    )
    second = AnilistClient(
        anilist_token="token",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
        rate_limit=120,
    )

    assert first._request_limiter is not second._request_limiter
    assert first._request_limiter.rate == pytest.approx(2.0)
    assert second._request_limiter.rate == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_get_session_creates_and_reuses_client_session(
    monkeypatch: pytest.MonkeyPatch,
):
    """_get_session should build the aiohttp session with auth headers once."""
    created_headers: list[dict[str, str]] = []

    class DummySession:
        def __init__(self, *, headers: dict[str, str]):
            self.headers = headers
            self.closed = False
            created_headers.append(headers)

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "anibridge.providers.list.anilist.client.aiohttp.ClientSession",
        lambda *, headers: DummySession(headers=headers),
    )

    stub_client = AnilistClient(
        anilist_token="abc",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
    )

    session_one = await stub_client._get_session()
    assert created_headers[0]["Authorization"] == "Bearer abc"

    session_two = await stub_client._get_session()
    assert session_one is session_two

    cast(DummySession, session_one).closed = True
    session_three = await stub_client._get_session()
    assert session_three is not session_one


@pytest.mark.asyncio
async def test_close_ignores_already_closed_session():
    """Close should only attempt to run when the session is still open."""

    class DummySession:
        def __init__(self) -> None:
            self.closed = False
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            self.closed = True

    stub_client = AnilistClient(
        anilist_token="abc",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
    )
    stub_client._session = cast(aiohttp.ClientSession, DummySession())

    await stub_client.close()
    await stub_client.close()  # second call should be a no-op

    assert cast(DummySession, stub_client._session).close_calls == 1


@pytest.mark.asyncio
async def test_initialize_parses_user_timezone_offset(client: AnilistClient):
    """Client.initialize should parse the user's timezone string into a tzinfo.

    It should accept timezone offsets both with and without a leading sign.
    """

    # timezone without a sign should be treated as positive
    async def first_user() -> User:
        return User.model_construct(
            id=1, name="tz", options=UserOptions(timezone="02:30")
        )

    async def noop() -> None:
        pass

    client.get_user = first_user  # ty:ignore[invalid-assignment]
    client._fetch_list_collection = noop  # ty:ignore[invalid-assignment]
    await client.initialize()

    assert client.user_timezone.utcoffset(None) == timedelta(hours=2, minutes=30)

    # negative offsets should also be parsed correctly
    async def second_user() -> User:
        return User.model_construct(
            id=1, name="tz", options=UserOptions(timezone="-05:00")
        )

    client.get_user = second_user  # ty:ignore[invalid-assignment]
    await client.initialize()

    assert client.user_timezone.utcoffset(None) == timedelta(hours=-5)


@pytest.mark.asyncio
async def test_get_user_returns_viewer_payload(client: AnilistClient):
    """Get_user should deserialize the Viewer response into a User model."""

    async def fake_request(*_args: Any, **_kwargs: Any) -> dict:
        return {
            "data": {
                "Viewer": {
                    "id": 42,
                    "name": "Viewer",
                    "mediaListOptions": None,
                }
            }
        }

    client._make_request = fake_request  #  ty:ignore[invalid-assignment]

    viewer = await client.get_user()

    assert viewer.id == 42
    assert viewer.name == "Viewer"


@pytest.mark.asyncio
async def test_get_anime_prefers_cached_entry(client: AnilistClient, media_factory):
    """get_anime should avoid the network when a cached entry exists."""
    cached_media: Media = media_factory(999, "cached show")
    client._list_cache[cached_media.id] = cached_media

    async def should_not_call(*_args: Any, **_kwargs: Any) -> dict:
        raise AssertionError("Network should not be called for cached anime")

    client._make_request = should_not_call  # ty:ignore[invalid-assignment]
    client._schedule_list_refresh = lambda: None  # ty:ignore[invalid-assignment]

    result = await client.get_anime(cached_media.id)

    assert result is cached_media


@pytest.mark.asyncio
async def test_batch_get_anime_fetches_missing_ids(
    client: AnilistClient, media_factory
):
    """batch_get_anime should mix cached entries with fetched ones."""
    cached_media: Media = media_factory(101, "cached")
    fetched_media: Media = media_factory(202, "fetched")
    client._list_cache[cached_media.id] = cached_media
    client._schedule_list_refresh = lambda: None  # ty:ignore[invalid-assignment]

    async def fake_request(query: str, variables: dict | None = None, **_: Any) -> dict:
        assert variables == {"ids": [fetched_media.id]}
        return {
            "data": {
                "Page": {
                    "media": [fetched_media.model_dump()],
                }
            }
        }

    client._make_request = fake_request  #  ty:ignore[invalid-assignment]

    results = await client.batch_get_anime([cached_media.id, fetched_media.id])

    assert cached_media in results
    assert fetched_media.id in client._list_cache
    assert any(media.id == fetched_media.id for media in results)


@pytest.mark.asyncio
async def test_batch_get_anime_returns_cached_when_all_present(
    client: AnilistClient, media_factory: Callable[[int, str], Media]
):
    """batch_get_anime should skip network work when everything is cached."""
    first = media_factory(10, "alpha")
    second = media_factory(20, "beta")
    client._list_cache[first.id] = first
    client._list_cache[second.id] = second
    client._schedule_list_refresh = lambda: None  # ty:ignore[invalid-assignment]

    async def should_not_call(*_args: Any, **_kwargs: Any) -> dict:
        raise AssertionError("Network should not be called when all entries are cached")

    client._make_request = should_not_call  # ty:ignore[invalid-assignment]

    results = await client.batch_get_anime([first.id, second.id])

    assert results == [first, second]


@pytest.mark.asyncio
async def test_batch_get_anime_handles_empty_input(client: AnilistClient):
    """batch_get_anime should immediately return when no ids are provided."""
    assert await client.batch_get_anime([]) == []


@pytest.mark.asyncio
async def test_search_anime_filters_episode_and_status(
    client: AnilistClient, media_factory: Callable[[int, str], Media]
):
    """search_anime should honor both release status and episode filters."""
    releasing: Media = media_factory(1, "airing")
    releasing.status = MediaStatus.RELEASING
    releasing.episodes = None

    finished_matching: Media = media_factory(2, "finished")
    finished_matching.status = MediaStatus.FINISHED
    finished_matching.episodes = 24

    skipped: Media = media_factory(3, "skipped")
    skipped.status = MediaStatus.FINISHED
    skipped.episodes = 12

    async def fake_search(*_args: Any, **_kwargs: Any) -> list[Media]:
        return [releasing, finished_matching, skipped]

    client._search_anime = fake_search  # ty:ignore[invalid-assignment]

    results: list[Media] = []
    async for media in client.search_anime("foo", is_movie=None, episodes=24):
        results.append(media)

    assert results == [releasing, finished_matching]


@pytest.mark.asyncio
async def test_search_anime_without_episode_filter_returns_all(
    client: AnilistClient, media_factory: Callable[[int, str], Media]
):
    """search_anime should yield every result when an episode filter is absent."""
    finished: Media = media_factory(4, "finished")
    finished.status = MediaStatus.FINISHED
    finished.episodes = 12

    async def fake_search(*_args: Any, **_kwargs: Any) -> list[Media]:
        return [finished]

    client._search_anime = fake_search  # ty:ignore[invalid-assignment]

    results = [media async for media in client.search_anime("foo", is_movie=False)]

    assert results == [finished]


def _build_saved_entry(media_id: int, title: str) -> MediaListWithMedia:
    """Helper that fabricates a MediaListWithMedia payload."""
    media_meta = MediaWithoutList(
        id=media_id,
        format=MediaFormat.TV,
        status=MediaStatus.RELEASING,
        title=MediaTitle(romaji=title, english=title.title()),
        episodes=24,
    )
    return MediaListWithMedia(
        id=media_id * 10,
        user_id=1,
        media_id=media_id,
        status=MediaListStatus.CURRENT,
        progress=5,
        repeat=0,
        media=media_meta,
    )


@pytest.mark.asyncio
async def test_update_anime_entry_caches_saved_media(client: AnilistClient):
    """Updating an entry should refresh the list cache with the response payload."""
    entry = MediaList(id=10, user_id=1, media_id=777, status=MediaListStatus.CURRENT)
    saved_entry = _build_saved_entry(entry.media_id, "cache me")

    async def fake_request(*_args: Any, **_kwargs: Any) -> dict:
        return {"data": {"SaveMediaListEntry": saved_entry.model_dump()}}

    client._make_request = fake_request  #  ty:ignore[invalid-assignment]

    await client.update_anime_entry(entry)

    cached = client._list_cache[entry.media_id]
    assert cached.media_list_entry is not None
    assert cached.media_list_entry.progress == 5


@pytest.mark.asyncio
async def test_update_anime_entry_invalidates_cached_backup(
    client: AnilistClient,
):
    """Updating an entry should force backup_anilist to fetch fresh list data."""
    client.user = User.model_construct(id=1, name="Backup Tester")
    entry = MediaList(
        id=10,
        user_id=1,
        media_id=777,
        status=MediaListStatus.CURRENT,
        progress=5,
    )
    collection_calls = 0
    state = {"progress": 1}

    async def fake_request(query: str, *_args: Any, **_kwargs: Any) -> dict:
        nonlocal collection_calls
        if "MediaListCollection" in query:
            collection_calls += 1
            return {
                "data": {
                    "MediaListCollection": {
                        "user": {"id": 1, "name": "Backup Tester"},
                        "lists": [
                            {
                                "entries": [
                                    {
                                        "id": 10,
                                        "userId": 1,
                                        "mediaId": 777,
                                        "status": "CURRENT",
                                        "progress": state["progress"],
                                        "media": {
                                            "id": 777,
                                            "format": "TV",
                                            "status": "RELEASING",
                                        },
                                    }
                                ],
                                "name": "Watching",
                                "isCustomList": False,
                                "status": "CURRENT",
                            }
                        ],
                        "hasNextChunk": False,
                    }
                }
            }
        if "SaveMediaListEntry" in query:
            state["progress"] = 5
            saved_entry = _build_saved_entry(entry.media_id, "cache me")
            return {"data": {"SaveMediaListEntry": saved_entry.model_dump()}}
        raise AssertionError(query)

    client._make_request = fake_request  #  ty:ignore[invalid-assignment]

    first_backup = json.loads(await client.backup_anilist())
    await client.update_anime_entry(entry)
    second_backup = json.loads(await client.backup_anilist())

    assert first_backup["lists"][0]["entries"][0]["progress"] == 1
    assert second_backup["lists"][0]["entries"][0]["progress"] == 5
    assert collection_calls == 2


@pytest.mark.asyncio
async def test_update_anime_entry_clears_search_cache(client: AnilistClient):
    """Updating an entry should invalidate any cached search results."""

    class CachedSearch:
        def __init__(self) -> None:
            self.clear_calls = 0

        def cache_clear(self) -> None:
            self.clear_calls += 1

    entry = MediaList(id=10, user_id=1, media_id=777, status=MediaListStatus.CURRENT)
    saved_entry = _build_saved_entry(entry.media_id, "cache me")
    fake_search = CachedSearch()

    async def fake_request(*_args: Any, **_kwargs: Any) -> dict:
        return {"data": {"SaveMediaListEntry": saved_entry.model_dump()}}

    client._search_anime = cast(Any, fake_search)
    client._make_request = fake_request  #  ty:ignore[invalid-assignment]

    await client.update_anime_entry(entry)

    assert fake_search.clear_calls == 1


@pytest.mark.asyncio
async def test_delete_anime_entry_requires_user(client: AnilistClient):
    """Deleting without a user context should raise a client error."""
    with pytest.raises(aiohttp.ClientError):
        await client.delete_anime_entry(entry_id=1, media_id=1)


@pytest.mark.asyncio
async def test_delete_anime_entry_removes_cache(
    client: AnilistClient, media_factory: Callable[[int, str], Media]
):
    """Successful deletes should purge cached entries by media id."""
    media = media_factory(303, "delete me")
    client.user = User.model_construct(id=1, name="Tester")
    client._list_cache[media.id] = media
    assert media.media_list_entry is not None, (
        "Precondition: media must have list entry"
    )

    async def fake_request(*_: Any, **__: Any) -> dict:
        return {"data": {"DeleteMediaListEntry": {"deleted": True}}}

    client._make_request = fake_request  #  ty:ignore[invalid-assignment]

    deleted = await client.delete_anime_entry(media.media_list_entry.id, media.id)

    assert deleted is True
    assert media.id not in client._list_cache


@pytest.mark.asyncio
async def test_backup_anilist_returns_sanitized_json(client: AnilistClient):
    """backup_anilist should strip media metadata and populate the cache."""
    user = User.model_construct(id=1, name="Backup Tester")
    client.user = user

    saved_entry = _build_saved_entry(404, "backup show")
    list_with_media = MediaListGroupWithMedia.model_construct(
        entries=[saved_entry],
        name="Plan to Watch",
        is_custom_list=False,
        status=MediaListStatus.PLANNING,
    )
    custom_group = MediaListGroupWithMedia.model_construct(
        entries=[_build_saved_entry(999, "custom skip")],
        name="Custom",
        is_custom_list=True,
        status=MediaListStatus.CURRENT,
    )
    collection = MediaListCollectionWithMedia.model_construct(
        user=user,
        lists=[list_with_media, custom_group],
        has_next_chunk=False,
    )

    async def fake_request(
        _query: str, variables: dict | None = None, **_: Any
    ) -> dict:
        assert variables == {"userId": 1, "type": "ANIME", "chunk": 0}
        return {"data": {"MediaListCollection": collection.model_dump()}}

    client._make_request = fake_request  #  ty:ignore[invalid-assignment]

    backup_payload = await client.backup_anilist()
    parsed = json.loads(backup_payload)

    assert parsed["user"]["id"] == 1
    assert parsed["lists"][0]["entries"][0]["mediaId"] == 404
    assert "media" not in parsed["lists"][0]["entries"][0]
    assert len(parsed["lists"]) == 1
    assert 404 in client._list_cache


@pytest.mark.asyncio
async def test_fetch_list_collection_keeps_media_cache_metadata_only(
    client: AnilistClient, media_factory: Callable[[int, str], Media]
):
    """A list refresh should not resurrect stale list-entry state from _media_cache."""
    user = User.model_construct(id=1, name="Refresh Tester")
    client.user = user
    media = media_factory(404, "stale list entry")
    client._remember(media)

    async def fake_request(*_args: Any, **_kwargs: Any) -> dict:
        return {
            "data": {
                "MediaListCollection": {
                    "user": {"id": 1, "name": "Refresh Tester"},
                    "lists": [],
                    "hasNextChunk": False,
                }
            }
        }

    client._make_request = fake_request  #  ty:ignore[invalid-assignment]

    await client._fetch_list_collection()

    cached = client._cached(media.id)
    assert cached is not None
    assert cached.media_list_entry is None


@pytest.mark.asyncio
async def test_backup_anilist_requires_user(client: AnilistClient):
    """backup_anilist should raise when invoked without an authenticated user."""
    with pytest.raises(aiohttp.ClientError):
        await client.backup_anilist()


@pytest.mark.asyncio
async def test_restore_anilist_invokes_batch_update(client: AnilistClient):
    """restore_anilist should fan out the parsed entries to batch updates."""
    entry = MediaList(id=11, user_id=1, media_id=222, status=MediaListStatus.CURRENT)
    group = MediaListGroup.model_construct(
        entries=[entry],
        name="Watching",
        status=MediaListStatus.CURRENT,
    )
    collection = MediaListCollection.model_construct(
        user=None,
        lists=[group],
        has_next_chunk=False,
    )
    backup = collection.model_dump_json()

    recorded_entries: list[MediaList] | None = None

    async def fake_batch_update(entries: list[MediaList]) -> set[int]:
        nonlocal recorded_entries
        recorded_entries = entries
        return {entry.media_id for entry in entries}

    client.batch_update_anime_entries = fake_batch_update  # ty:ignore[invalid-assignment]

    await client.restore_anilist(backup)

    assert recorded_entries is not None
    assert isinstance(recorded_entries[0], MediaList)


@pytest.mark.asyncio
async def test_media_list_entry_to_media_merges_metadata(client: AnilistClient):
    """_to_media should combine list and media fields."""
    saved_entry = _build_saved_entry(515, "merge target")
    media = client._to_media(saved_entry)

    assert media.media_list_entry is not None
    assert media.media_list_entry.media_id == 515
    assert media.title is not None
    assert media.title.romaji == "merge target"


class _ResponseContext:
    """Async context manager that mimics aiohttp responses for _make_request."""

    def __init__(
        self,
        status: int,
        payload: dict | None = None,
        headers: dict | None = None,
        *,
        raise_error: bool = False,
    ):
        self.status = status
        self._payload = payload or {"data": {}}
        self.headers = headers or {}
        self._raise_error = raise_error

    async def __aenter__(self) -> _ResponseContext:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self._raise_error:
            raise aiohttp.ClientResponseError(
                request_info=cast(Any, None),
                history=(),
                status=self.status,
                message="boom",
            )

    async def json(self) -> dict:
        return self._payload

    async def text(self) -> str:
        return "error-text"


class _FakeSession:
    """Simple aiohttp session stub that returns preloaded responses."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.closed = False

    def post(self, *_args, **_kwargs):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_make_request_retries_rate_limit(monkeypatch: pytest.MonkeyPatch):
    """Rate limit responses should trigger a retry after sleeping."""
    session = _FakeSession(
        [
            _ResponseContext(429, headers={"Retry-After": "0"}),
            _ResponseContext(200, payload={"data": {"ok": True}}),
        ]
    )
    client = AnilistClient(
        anilist_token="token",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
    )

    async def fake_get_session() -> _FakeSession:
        return session

    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1

    client._get_session = fake_get_session  # ty:ignore[invalid-assignment]
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await client._make_request("query")

    assert result["data"]["ok"] is True
    assert sleep_calls >= 1


@pytest.mark.asyncio
async def test_make_request_retries_bad_gateway(monkeypatch: pytest.MonkeyPatch):
    """502 responses should also be retried using the same session."""
    session = _FakeSession(
        [
            _ResponseContext(502),
            _ResponseContext(200, payload={"data": {"ok": 2}}),
        ]
    )
    client = AnilistClient(
        anilist_token="token",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
    )

    async def fake_get_session() -> _FakeSession:
        return session

    async def fake_sleep(_seconds: float) -> None:
        return None

    client._get_session = fake_get_session  # ty:ignore[invalid-assignment]
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await client._make_request("query")

    assert result["data"]["ok"] == 2


@pytest.mark.asyncio
async def test_make_request_recovers_from_client_error(monkeypatch: pytest.MonkeyPatch):
    """Unexpected client exceptions should be retried up to the limit."""
    session = _FakeSession(
        [
            aiohttp.ClientError("boom"),
            _ResponseContext(200, payload={"data": {"ok": 3}}),
        ]
    )
    client = AnilistClient(
        anilist_token="token",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
    )

    async def fake_get_session() -> _FakeSession:
        return session

    async def fake_sleep(_seconds: float) -> None:
        return None

    client._get_session = fake_get_session  # ty:ignore[invalid-assignment]
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = await client._make_request("query")

    assert result["data"]["ok"] == 3


@pytest.mark.asyncio
async def test_make_request_raises_after_three_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    """_make_request should raise once the retry budget is exhausted."""
    session = _FakeSession([cast(object, aiohttp.ClientError("boom"))] * 4)
    client = AnilistClient(
        anilist_token="token",
        logger=cast(ProviderLogger, logging.getLogger("tests.client")),
    )

    async def fake_get_session() -> _FakeSession:
        return session

    async def fake_sleep(_seconds: float) -> None:
        return None

    client._get_session = fake_get_session  # ty:ignore[invalid-assignment]
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(aiohttp.ClientError):
        await client._make_request("query")


def _media_list(media_id: int) -> MediaList:
    """Return a basic MediaList payload for batch update tests."""
    return MediaList(
        id=media_id * 10,
        user_id=1,
        media_id=media_id,
        status=MediaListStatus.CURRENT,
        progress=1,
        repeat=0,
    )


@pytest.mark.asyncio
async def test_batch_update_anime_entries_handles_empty_input(client: AnilistClient):
    """batch_update_anime_entries should early-exit when given no entries."""
    assert await client.batch_update_anime_entries([]) == set()


@pytest.mark.asyncio
async def test_batch_update_anime_entries_updates_cache(client: AnilistClient):
    """Batch updates should process entries in multiple chunks and cache results."""
    entries = [_media_list(i) for i in range(1, 12)]
    calls = 0

    async def fake_request(
        query: str, variables: str | None = None, retry_count: int = 0, **_: Any
    ) -> dict:
        nonlocal calls
        start = calls * 10
        chunk = entries[start : start + 10]
        payload: dict[str, Any] = {}
        for idx, entry in enumerate(chunk):
            payload[f"m{idx}"] = {
                "id": entry.id,
                "userId": entry.user_id,
                "mediaId": entry.media_id,
                "status": entry.status.value if entry.status else None,
                "progress": entry.progress,
                "repeat": entry.repeat,
                "media": {
                    "id": entry.media_id,
                    "format": MediaFormat.TV,
                    "status": MediaStatus.RELEASING,
                },
            }
        calls += 1
        return {"data": payload}

    client._make_request = fake_request  # type: ignore

    await client.batch_update_anime_entries(entries)

    assert calls == 2
    assert len(client._list_cache) == len(entries)


@pytest.mark.asyncio
async def test_get_anime_fetches_from_api_when_not_cached(
    client: AnilistClient, media_factory: Callable[[int, str], Media]
):
    """get_anime should call out to AniList when the cache is missing the id."""
    media = media_factory(505, "api show")
    client._schedule_list_refresh = lambda: None  # ty:ignore[invalid-assignment]

    async def fake_request(*_args: Any, **_kwargs: Any) -> dict:
        return {"data": {"Media": media.model_dump()}}

    client._make_request = fake_request  # ty:ignore[invalid-assignment]

    result = await client.get_anime(media.id)

    assert result.id == media.id
    assert client._list_cache[media.id].id == media.id


@pytest.mark.asyncio
async def test_batch_get_anime_mixed_cache_uses_network(
    client: AnilistClient, media_factory: Callable[[int, str], Media]
):
    """batch_get_anime should fallback to the API for missing ids."""
    cached = media_factory(1, "cached")
    missing = media_factory(2, "missing")
    client._list_cache[cached.id] = cached
    client._schedule_list_refresh = lambda: None  # ty:ignore[invalid-assignment]

    async def fake_request(
        query: str, variables: dict | None = None, retry_count: int = 0, **_: Any
    ) -> dict:
        assert variables == {"ids": [missing.id]}
        return {
            "data": {
                "Page": {
                    "media": [missing.model_dump()],
                }
            }
        }

    client._make_request = fake_request  # type: ignore

    results = await client.batch_get_anime([cached.id, missing.id])

    assert set(media.id for media in results) == {cached.id, missing.id}
    assert missing.id in client._list_cache


@pytest.mark.asyncio
async def test__search_anime_uses_movie_formats(client: AnilistClient):
    """_search_anime should restrict formats to movies when requested."""
    captured: dict[str, Any] = {}

    async def fake_request(
        query: str, variables: dict | None = None, retry_count: int = 0, **_: Any
    ) -> dict:
        captured.update(variables or {})
        return {"data": {"Page": {"media": []}}}

    client._make_request = fake_request  # type: ignore

    await client._search_anime("title", is_movie=True, limit=5)

    assert captured["formats"] == [MediaFormat.MOVIE, MediaFormat.SPECIAL]


@pytest.mark.asyncio
async def test__search_anime_uses_show_formats(client: AnilistClient):
    """_search_anime should restrict formats to shows when requested."""
    captured: dict[str, Any] = {}

    async def fake_request(
        query: str, variables: dict | None = None, retry_count: int = 0, **_: Any
    ) -> dict:
        captured.update(variables or {})
        return {"data": {"Page": {"media": []}}}

    client._make_request = fake_request  # type: ignore

    await client._search_anime("title", is_movie=False, limit=3)

    assert set(captured["formats"]) == {
        MediaFormat.TV,
        MediaFormat.TV_SHORT,
        MediaFormat.ONA,
        MediaFormat.OVA,
    }
