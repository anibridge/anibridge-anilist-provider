"""AniList Client."""

import asyncio
import contextlib
import importlib.metadata
import json
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, timedelta, timezone, tzinfo
from logging import Logger
from typing import Any, ClassVar, cast

import aiohttp
import msgspec
from anibridge.utils.cache import TTLDict, ttl_cache
from anibridge.utils.limiter import Limiter

from anibridge.providers.anilist.models import (
    Media,
    MediaFormat,
    MediaList,
    MediaListCollection,
    MediaListCollectionWithMedia,
    MediaListGroup,
    MediaListStatus,
    MediaListWithMedia,
    MediaStatus,
    User,
)

__all__ = ["AnilistClient"]

_ANILIST_PACKAGE_NAME = "anibridge-anilist-provider"

global_anilist_limiter = Limiter(30 / 60, capacity=4)


def _without_none(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


class AnilistClient:
    """Client for interacting with the AniList GraphQL API.

    This client provides methods to interact with the AniList GraphQL API, including
    searching for anime, updating user lists, and managing anime entries. It implements
    rate limiting and local caching to optimize API usage.
    """

    API_URL: ClassVar[str] = "https://graphql.anilist.co"
    _BATCH_SIZE: ClassVar[int] = 10
    _LIST_REFRESH_DEBOUNCE_SECONDS: ClassVar[float] = 60.0
    _LIST_REFRESH_MAX_STALENESS_SECONDS: ClassVar[float] = 1800.0

    def __init__(
        self,
        anilist_token: str,
        logger: Logger,
        rate_limit: int | None = None,
    ) -> None:
        """Initialize the AniList client."""
        self.anilist_token = anilist_token
        self.rate_limit = rate_limit
        self.log = logger
        self._session: aiohttp.ClientSession | None = None

        if self.rate_limit is None:
            self.log.debug(
                "Using shared global AniList rate limiter with %s requests per minute",
                global_anilist_limiter.rate * 60,
            )
            self._request_limiter = global_anilist_limiter
        else:
            self.log.debug(
                "Using local AniList rate limiter with %s requests per minute",
                self.rate_limit,
            )
            self._request_limiter = Limiter(self.rate_limit / 60, capacity=1)

        self.user: User | None = None
        self.user_timezone: tzinfo = UTC

        self._bg_task: asyncio.Task[None] | None = None
        self._cache_epoch = 0
        self._list_cache_dirty = False
        self._last_list_refresh_at = time.monotonic()
        self._list_cache: dict[int, Media] = {}
        self._media_cache: TTLDict[int, Media] = TTLDict(ttl=43200, maxsize=2048)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": self._user_agent(),
            }
            if self.anilist_token:
                headers["Authorization"] = f"Bearer {self.anilist_token}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if (task := self._bg_task) and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._session and not self._session.closed:
            await self._session.close()

    def clear_cache(self) -> None:
        """Clear in-memory caches for user list and general media lookups."""
        self._list_cache.clear()
        self._media_cache.clear()
        self._list_cache_dirty = False
        self._last_list_refresh_at = time.monotonic()
        self._invalidate_cached_views()

    def _invalidate_cached_views(
        self, *, clear_list_collection_cache: bool = True
    ) -> None:
        """Invalidate derived cached views after list-state changes."""
        self._cache_epoch += 1
        if (task := self._bg_task) and not task.done():
            task.cancel()
        self._bg_task = None
        if clear_list_collection_cache:
            with contextlib.suppress(AttributeError):
                self._fetch_list_collection.cache_clear()
        with contextlib.suppress(AttributeError):
            self._search_anime.cache_clear()

    @staticmethod
    def _user_agent() -> str:
        """Return the package-specific AniList user agent."""
        return (
            f"{_ANILIST_PACKAGE_NAME}/"
            f"{importlib.metadata.version(_ANILIST_PACKAGE_NAME)}"
        )

    async def initialize(self):
        """Initialize the client by getting user info and prefetching entries."""
        self.clear_cache()
        self.user = await self.get_user()

        # Timezone in "-?HH:MM" format
        offset_str = self.user.options.timezone if self.user.options else None
        if offset_str:
            if offset_str[0] not in "+-":
                offset_str = "+" + offset_str
            sign = 1 if offset_str[0] == "+" else -1
            hours, minutes = map(int, offset_str[1:].split(":"))
            self.user_timezone = timezone(
                sign * timedelta(hours=hours, minutes=minutes)
            )

        await self._fetch_list_collection()

    def _cached(self, anilist_id: int) -> Media | None:
        """Return media from user-list cache or general TTL cache."""
        return self._list_cache.get(anilist_id) or self._media_cache.get(anilist_id)

    def _remember(self, media: Media) -> None:
        """Store media in shared caches."""
        # Keep the long-lived TTL cache free of user-specific list state.
        self._media_cache[media.id] = msgspec.convert(
            {**msgspec.to_builtins(media), "mediaListEntry": None},
            type=Media,
        )
        if media.media_list_entry is None:
            self._list_cache.pop(media.id, None)
        else:
            self._list_cache[media.id] = media

    def _schedule_list_refresh(self) -> None:
        """Schedule a background refresh when list state is dirty or stale."""
        stale_for = time.monotonic() - self._last_list_refresh_at
        is_stale = stale_for >= self._LIST_REFRESH_MAX_STALENESS_SECONDS

        if is_stale:
            self._enqueue_list_refresh(debounce_seconds=0.0)
            return

        if self._list_cache_dirty:
            self._enqueue_list_refresh(
                debounce_seconds=self._LIST_REFRESH_DEBOUNCE_SECONDS,
            )

    def _mark_list_cache_dirty(self) -> None:
        """Mark list state dirty and debounce a single full refresh."""
        self._list_cache_dirty = True
        self._schedule_list_refresh()

    def _enqueue_list_refresh(self, *, debounce_seconds: float) -> None:
        """Queue one background list refresh, optionally debounced."""
        if (task := self._bg_task) and not task.done():
            task.cancel()
            self._bg_task = None

        async def _refresh() -> None:
            if debounce_seconds > 0:
                await asyncio.sleep(debounce_seconds)
            with contextlib.suppress(AttributeError):
                self._fetch_list_collection.cache_clear()
            await self._fetch_list_collection()

        def _on_done(t: asyncio.Task[None]) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                self.log.warning("User-list cache refresh failed", exc_info=exc)

        self._bg_task = task = asyncio.create_task(_refresh())
        task.add_done_callback(_on_done)

    async def get_user(self) -> User:
        """Retrieves the authenticated user's information from AniList."""
        query = f"""
        query {{
            Viewer {{
                {User.model_dump_graphql()}
            }}
        }}
        """
        response = await self._make_request(query)
        return msgspec.convert(response["data"]["Viewer"], type=User)

    async def update_anime_entry(self, entry: MediaList) -> None:
        """Updates an anime entry on the authenticated user's list."""
        query = f"""
        mutation (
            $mediaId: Int, $status: MediaListStatus, $score: Float, $progress: Int,
            $repeat: Int, $notes: String, $startedAt: FuzzyDateInput,
            $completedAt: FuzzyDateInput
        ) {{
            SaveMediaListEntry(
                mediaId: $mediaId, status: $status, score: $score, progress: $progress,
                repeat: $repeat, notes: $notes, startedAt: $startedAt,
                completedAt: $completedAt
            ) {{
                {MediaListWithMedia.model_dump_graphql()}
            }}
        }}
        """
        response = await self._make_request(
            query,
            cast(
                dict[str, Any],
                _without_none(msgspec.json.decode(msgspec.json.encode(entry))),
            ),
        )
        saved = msgspec.convert(
            response["data"]["SaveMediaListEntry"], type=MediaListWithMedia
        )
        self._list_cache[entry.media_id] = self._to_media(saved)
        self._media_cache.pop(entry.media_id, None)
        self._invalidate_cached_views(clear_list_collection_cache=False)
        self._mark_list_cache_dirty()

    async def delete_anime_entry(self, entry_id: int, media_id: int) -> bool:
        """Deletes an anime entry from the authenticated user's list."""
        if not self.user:
            raise aiohttp.ClientError("User information is required for deletions")

        query = """
        mutation ($id: Int) {
            DeleteMediaListEntry(id: $id) {
                deleted
            }
        }
        """
        variables = MediaList(id=entry_id, media_id=media_id, user_id=self.user.id)
        variables = cast(
            dict[str, Any],
            _without_none(msgspec.json.decode(msgspec.json.encode(variables))),
        )

        response = await self._make_request(query, variables)
        self._list_cache.pop(media_id, None)
        self._media_cache.pop(media_id, None)
        self._invalidate_cached_views(clear_list_collection_cache=False)
        self._mark_list_cache_dirty()
        return response["data"]["DeleteMediaListEntry"]["deleted"]

    async def batch_update_anime_entries(self, entries: list[MediaList]) -> set[int]:
        """Updates multiple anime entries on the authenticated user's list.

        Sends a batch mutation to modify multiple existing anime entries in the user's
        list. Processes entries in batches of 10 to avoid overwhelming the API.
        """
        if not entries:
            return set()

        fields = [
            ("mediaId", "Int"),
            ("status", "MediaListStatus"),
            ("score", "Float"),
            ("progress", "Int"),
            ("repeat", "Int"),
            ("notes", "String"),
            ("startedAt", "FuzzyDateInput"),
            ("completedAt", "FuzzyDateInput"),
        ]

        updated: set[int] = set()

        for i in range(0, len(entries), self._BATCH_SIZE):
            batch = entries[i : i + self._BATCH_SIZE]
            self.log.debug(
                "Updating batch of anime entries $${anilist_id: %s}$$",
                [m.media_id for m in batch],
            )
            by_id = {e.media_id: e for e in batch}

            var_decls, mutations, variables = [], [], {}
            for j, e in enumerate(batch):
                var_decls.extend(f"${f}{j}: {t}" for f, t in fields)
                mutations.append(f"""
                    m{j}: SaveMediaListEntry(
                        mediaId: $mediaId{j}, status: $status{j}, score: $score{j},
                        progress: $progress{j}, repeat: $repeat{j}, notes: $notes{j},
                        startedAt: $startedAt{j}, completedAt: $completedAt{j}
                    ) {{ {MediaListWithMedia.model_dump_graphql()} }}
                """)
                encoded_entry = cast(
                    dict[str, Any],
                    _without_none(msgspec.json.decode(msgspec.json.encode(e))),
                )
                for k, v in encoded_entry.items():
                    variables[f"{k}{j}"] = v

            query = f"""
            mutation BatchUpdateEntries({", ".join(var_decls)}) {{
                {chr(10).join(mutations)}
            }}
            """

            try:
                response: dict[str, dict[str, dict]] = await self._make_request(
                    query, variables
                )
            except aiohttp.ClientError as exc:
                self.log.warning(
                    "Batch update failed; falling back to per-entry updates",
                    exc_info=exc,
                )
                updated.update(await self._fallback_update(batch))
                continue

            batch_ok: set[int] = set()
            for data in response.get("data", {}).values():
                if not data or "mediaId" not in data:
                    continue
                mid = data["mediaId"]
                self._list_cache[mid] = self._to_media(
                    msgspec.convert(data, type=MediaListWithMedia)
                )
                batch_ok.add(mid)

            updated.update(batch_ok)
            missing = [by_id[mid] for mid in set(by_id) - batch_ok]
            if missing:
                updated.update(await self._fallback_update(missing))

        for media_id in updated:
            self._media_cache.pop(media_id, None)
        self._invalidate_cached_views(clear_list_collection_cache=False)
        if updated:
            self._mark_list_cache_dirty()
        return updated

    async def _fallback_update(self, entries: list[MediaList]) -> set[int]:
        """Update entries one-by-one, swallowing individual failures."""
        ok: set[int] = set()
        for e in entries:
            try:
                await self.update_anime_entry(e)
                ok.add(e.media_id)
            except aiohttp.ClientError as exc:
                self.log.warning(
                    "Failed to update AniList entry %s", e.media_id, exc_info=exc
                )
        return ok

    async def search_anime(
        self,
        search_str: str,
        is_movie: bool | None,
        episodes: int | None = None,
        limit: int = 10,
    ) -> AsyncIterator[Media]:
        """Search for anime on AniList with filtering capabilities."""
        kind = "all" if is_movie is None else ("movie" if is_movie else "show")
        self.log.debug(
            "Searching for %s with title $$'%s'$$ that is releasing and has "
            "%s episodes",
            kind,
            search_str,
            episodes or "unknown",
        )
        for m in await self._search_anime(search_str, is_movie, limit):
            if (
                m.status == MediaStatus.RELEASING
                or m.episodes == episodes
                or not episodes
            ):
                yield m

    @ttl_cache(ttl=300, maxsize=128)
    async def _search_anime(
        self, search_str: str, is_movie: bool | None, limit: int = 10
    ) -> list[Media]:
        """Cached helper function for anime searches."""
        formats = (
            [MediaFormat.MOVIE, MediaFormat.SPECIAL]
            if is_movie is True
            else [
                MediaFormat.TV,
                MediaFormat.TV_SHORT,
                MediaFormat.ONA,
                MediaFormat.OVA,
            ]
            if is_movie is False
            else [
                MediaFormat.MOVIE,
                MediaFormat.SPECIAL,
                MediaFormat.TV,
                MediaFormat.TV_SHORT,
                MediaFormat.ONA,
                MediaFormat.OVA,
            ]
        )
        query = f"""
        query ($search: String, $formats: [MediaFormat], $limit: Int) {{
            Page(perPage: $limit) {{
                media(search: $search, type: ANIME, format_in: $formats) {{
                    {Media.model_dump_graphql()}
                }}
            }}
        }}
        """
        response = await self._make_request(
            query, {"search": search_str, "formats": formats, "limit": limit}
        )
        return [
            msgspec.convert(m, type=Media) for m in response["data"]["Page"]["media"]
        ]

    async def get_anime(self, anilist_id: int) -> Media:
        """Retrieves detailed information about a specific anime.

        Attempts to fetch anime data from local cache first, falling back to
        an API request if not found in cache.
        """
        self._schedule_list_refresh()
        return self._cached(anilist_id) or await self._fetch_anime(anilist_id)

    async def batch_get_anime(self, anilist_ids: list[int]) -> list[Media]:
        """Retrieves detailed information about a list of anime.

        Attempts to fetch anime data from local cache first, falling back to
        batch API requests for entries not found in cache. Processes requests
        in batches of 10 to avoid overwhelming the API.
        """
        if not anilist_ids:
            return []

        self._schedule_list_refresh()

        cached_ids = [mid for mid in anilist_ids if mid in self._list_cache]
        if cached_ids:
            self.log.debug(
                "Pulling AniList data from local cache in batched mode "
                "$${anilist_ids: %s}$$",
                cached_ids,
            )

        missing = [mid for mid in anilist_ids if self._cached(mid) is None]
        fetched = await self._batch_fetch_anime(missing) if missing else {}

        result: list[Media] = []
        for aid in anilist_ids:
            hit = self._cached(aid) or fetched.get(aid)
            result.append(hit if hit is not None else await self._fetch_anime(aid))
        return result

    async def _fetch_anime(self, anilist_id: int) -> Media:
        """Fetch a media item from AniList API and populate caches."""
        query = f"""
        query ($id: Int) {{
            Media(id: $id, type: ANIME) {{
                {Media.model_dump_graphql()}
            }}
        }}
        """
        self.log.debug(
            "Pulling AniList data from API $${anilist_id: %s}$$",
            anilist_id,
        )
        response = await self._make_request(query, {"id": anilist_id})
        result = msgspec.convert(response["data"]["Media"], type=Media)
        self._remember(result)
        return result

    async def _batch_fetch_anime(self, anilist_ids: list[int]) -> dict[int, Media]:
        """Fetch media from AniList in batches for IDs missing in local list cache."""
        batch_size = 50
        fetched: dict[int, Media] = {}

        for i in range(0, len(anilist_ids), batch_size):
            batch = anilist_ids[i : i + batch_size]
            self.log.debug(
                "Pulling AniList data from API in batched mode $${anilist_ids: %s}$$",
                batch,
            )
            query = f"""
            query BatchGetAnime($ids: [Int]) {{
                Page(perPage: {len(batch)}) {{
                    media(id_in: $ids, type: ANIME) {{
                        {Media.model_dump_graphql()}
                    }}
                }}
            }}
            """
            try:
                response = await self._make_request(query, {"ids": batch})
            except aiohttp.ClientError as exc:
                self.log.warning("Batch fetch failed for IDs %s", batch, exc_info=exc)
                continue

            for raw in response.get("data", {}).get("Page", {}).get("media", []) or []:
                media = msgspec.convert(raw, type=Media)
                fetched[media.id] = media
                self._remember(media)

        return fetched

    @ttl_cache(ttl=3600, maxsize=1)
    async def _fetch_list_collection(self) -> None:
        """Fetch all non-custom AniList list entries and populate local cache."""
        if not self.user:
            raise aiohttp.ClientError("User information is required for fetching lists")

        query = f"""
        query MediaListCollection($userId: Int, $type: MediaType, $chunk: Int) {{
            MediaListCollection(userId: $userId, type: $type, chunk: $chunk) {{
                {MediaListCollectionWithMedia.model_dump_graphql()}
            }}
        }}
        """

        self.log.debug("Refreshing user list cache from AniList API")
        refresh_epoch = self._cache_epoch

        has_next_chunk = True
        refreshed_list_cache: dict[int, Media] = {}
        variables: dict[str, Any] = {
            "userId": self.user.id,
            "type": "ANIME",
            "chunk": 0,
        }

        while has_next_chunk:
            response = await self._make_request(query, variables)
            if refresh_epoch != self._cache_epoch:
                raise asyncio.CancelledError
            chunk = msgspec.convert(
                response["data"]["MediaListCollection"],
                type=MediaListCollectionWithMedia,
            )
            has_next_chunk = bool(chunk.has_next_chunk)
            variables["chunk"] += 1

            for li in chunk.lists:
                if li.is_custom_list:
                    continue
                for entry in li.entries:
                    refreshed_list_cache[entry.media_id] = self._to_media(entry)

        if refresh_epoch != self._cache_epoch:
            raise asyncio.CancelledError

        self._list_cache.clear()
        self._list_cache.update(refreshed_list_cache)
        self._list_cache_dirty = False
        self._last_list_refresh_at = time.monotonic()
        del refreshed_list_cache
        with contextlib.suppress(AttributeError):
            self._search_anime.cache_clear()

    async def backup_anilist(self) -> str:
        """Creates a JSON backup of the user's AniList data."""
        if not self.user:
            raise aiohttp.ClientError("User information is required for deletions")

        await self._fetch_list_collection()

        groups: dict[str | None, list[MediaList]] = defaultdict(list)
        for media in self._list_cache.values():
            if media.media_list_entry:
                key = (
                    media.media_list_entry.status.value
                    if media.media_list_entry.status
                    else None
                )
                groups[key].append(media.media_list_entry)

        sanitized = [
            MediaListGroup(
                entries=entries,
                status=MediaListStatus(status) if status else None,
            )
            for status, entries in groups.items()
        ]

        return msgspec.json.encode(
            MediaListCollection(user=self.user, lists=sanitized, has_next_chunk=False)
        ).decode()

    async def restore_anilist(self, backup: str) -> None:
        """Restores the user's AniList data from a JSON backup."""
        json_data = json.loads(backup)
        data = msgspec.convert(json_data, type=MediaListCollection)
        restored_entries = [entry for li in data.lists for entry in li.entries]
        restored_media_ids = {entry.media_id for entry in restored_entries}

        await self._fetch_list_collection()
        entries_to_delete = [
            media.media_list_entry
            for media in self._list_cache.values()
            if media.media_list_entry
            and media.media_list_entry.media_id not in restored_media_ids
        ]

        if restored_entries:
            await self.batch_update_anime_entries(restored_entries)

        for entry in entries_to_delete:
            await self.delete_anime_entry(entry.id, entry.media_id)

    @staticmethod
    def _to_media(entry: MediaListWithMedia) -> Media:
        """Converts a MediaListWithMedia object to a Media object."""
        entry_payload = dict(msgspec.to_builtins(entry))
        media_payload = entry_payload.pop("media")
        if media_payload is None:
            raise ValueError("AniList entry is missing media payload")

        return msgspec.convert(
            {
                **media_payload,
                "mediaListEntry": entry_payload,
            },
            type=Media,
        )

    async def _make_request(self, query: str, variables: dict | None = None) -> dict:
        """Make a rate-limited AniList GraphQL request with bounded retries."""
        max_attempts = 3
        non_retryable_statuses = {
            401: "Unauthorized API request (401). Verify your AniList token is valid.",
            403: (
                "Request forbidden (403). The API may be down or your token may "
                "lack permissions."
            ),
            404: "Not found (404). The requested resource might not exist.",
        }

        session = await self._get_session()
        for attempt in range(1, max_attempts + 1):
            try:
                await self._request_limiter.acquire(asynchronous=True)

                async with session.post(
                    self.API_URL, json={"query": query, "variables": variables or {}}
                ) as response:
                    if response.status in non_retryable_statuses:
                        clean_query = " ".join(query.split())
                        raise aiohttp.ClientError(
                            non_retryable_statuses[response.status]
                            + f"; query={clean_query}; variables={variables}"
                        )

                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After", "3")
                        delay = int(retry_after) if retry_after.isdigit() else 3

                        if attempt < max_attempts:
                            self.log.warning(
                                "AniList API rate limited (attempt %s/%s), retrying in "
                                "%ss",
                                attempt,
                                max_attempts,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue

                        raise aiohttp.ClientError(
                            f"AniList API rate limited (429) after {max_attempts} "
                            "attempts"
                        )

                    response.raise_for_status()
                    return await response.json()

            except (
                TimeoutError,
                aiohttp.ClientConnectionError,
                aiohttp.ClientResponseError,
            ) as exc:
                if attempt < max_attempts:
                    self.log.warning(
                        "Retrying AniList request (attempt %s/%s): %s",
                        attempt,
                        max_attempts,
                        exc,
                    )
                    await asyncio.sleep(1)
                    continue

                clean_query = " ".join(query.split())
                raise aiohttp.ClientError(
                    "AniList request failed after 3 attempts. "
                    f"error={exc.__class__.__name__}: {exc}; "
                    f"query={clean_query}; variables={variables}"
                ) from exc

        clean_query = " ".join(query.split())
        raise aiohttp.ClientError(
            f"AniList request failed unexpectedly; query={clean_query}; "
            f"variables={variables}"
        )
