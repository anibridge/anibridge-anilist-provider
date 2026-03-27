"""AniList Client."""

import asyncio
import contextlib
import importlib.metadata
import json
from collections.abc import AsyncIterator
from datetime import UTC, timedelta, timezone, tzinfo
from typing import Any

import aiohttp
from anibridge.utils.cache import TTLDict, ttl_cache
from anibridge.utils.limiter import Limiter
from anibridge.utils.types import ProviderLogger

from anibridge.providers.list.anilist.models import (
    Media,
    MediaFormat,
    MediaList,
    MediaListCollection,
    MediaListCollectionWithMedia,
    MediaListGroup,
    MediaListWithMedia,
    MediaStatus,
    User,
)

__all__ = ["AnilistClient"]

# Rate limit of 30 req/min with a burst capacity of 4
anilist_limiter = Limiter(rate=30 / 60, capacity=4)


class AnilistClient:
    """Client for interacting with the AniList GraphQL API.

    This client provides methods to interact with the AniList GraphQL API, including
    searching for anime, updating user lists, and managing anime entries. It implements
    rate limiting and local caching to optimize API usage.
    """

    API_URL = "https://graphql.anilist.co"

    def __init__(self, anilist_token: str, *, logger: ProviderLogger) -> None:
        """Initialize the AniList client.

        Args:
            anilist_token (str): Authentication token for AniList API.
            logger (ProviderLogger): Injected provider logger.
        """
        self.anilist_token = anilist_token
        self.log = logger
        self._session: aiohttp.ClientSession | None = None

        self.user: User | None = None
        self.user_timezone: tzinfo = UTC

        self._bg_task: asyncio.Task[MediaListCollectionWithMedia] | None = None
        self._cache_epoch = 0
        self._list_cache: dict[int, Media] = {}
        self._media_cache: TTLDict[int, Media] = TTLDict(ttl=43200)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "anibridge-anilist-provider/"
                + importlib.metadata.version("anibridge-anilist-provider"),
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
        self._invalidate_cached_views()

    def _invalidate_cached_views(self) -> None:
        """Invalidate derived cached views after list-state changes."""
        self._cache_epoch += 1
        if (task := self._bg_task) and not task.done():
            task.cancel()
        self._bg_task = None
        with contextlib.suppress(AttributeError):
            self._fetch_list_collection.cache_clear()
        with contextlib.suppress(AttributeError):
            self._search_anime.cache_clear()

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
        hit = self._list_cache.get(anilist_id) or self._media_cache.get(anilist_id)
        if hit:
            self.log.debug(f"Cache hit $${{anilist_id: {anilist_id}}}$$")
        return hit

    def _remember(self, media: Media) -> None:
        """Store media in shared caches."""
        # Keep the long-lived TTL cache free of user-specific list state.
        self._media_cache[media.id] = media.model_copy(
            update={"media_list_entry": None}
        )
        if media.media_list_entry is None:
            self._list_cache.pop(media.id, None)
        else:
            self._list_cache[media.id] = media

    def _schedule_list_refresh(self) -> None:
        """Schedule a background refresh when the user-list cache is stale.

        The @ttl_cache on _fetch_list_collection returns instantly from cache
        when fresh, so this is a no-op in the common case.
        """
        if (task := self._bg_task) and not task.done():
            return

        def _on_done(t: asyncio.Task[MediaListCollectionWithMedia]) -> None:
            if not t.cancelled() and (exc := t.exception()):
                self.log.warning("User-list cache refresh failed", exc_info=exc)

        self._bg_task = task = asyncio.create_task(self._fetch_list_collection())
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
        return User(**response["data"]["Viewer"])

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
            query, entry.model_dump_json(exclude_none=True)
        )
        saved = MediaListWithMedia(**response["data"]["SaveMediaListEntry"])
        self._list_cache[entry.media_id] = self._to_media(saved)
        self._media_cache.clear()
        self._invalidate_cached_views()

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
        variables = MediaList(
            id=entry_id, media_id=media_id, user_id=self.user.id
        ).model_dump_json(exclude_none=True)

        response = await self._make_request(query, variables)
        self._list_cache.pop(media_id, None)
        self._media_cache.pop(media_id, None)
        self._invalidate_cached_views()
        return response["data"]["DeleteMediaListEntry"]["deleted"]

    async def batch_update_anime_entries(self, entries: list[MediaList]) -> set[int]:
        """Updates multiple anime entries on the authenticated user's list.

        Sends a batch mutation to modify multiple existing anime entries in the user's
        list. Processes entries in batches of 10 to avoid overwhelming the API.
        """
        BATCH_SIZE = 10
        if not entries:
            return set()

        _FIELDS = [
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

        for i in range(0, len(entries), BATCH_SIZE):
            batch = entries[i : i + BATCH_SIZE]
            self.log.debug(
                f"Updating batch of anime entries "
                f"$${{anilist_id: {[m.media_id for m in batch]}}}$$"
            )
            by_id = {e.media_id: e for e in batch}

            var_decls, mutations, variables = [], [], {}
            for j, e in enumerate(batch):
                var_decls.extend(f"${f}{j}: {t}" for f, t in _FIELDS)
                mutations.append(f"""
                    m{j}: SaveMediaListEntry(
                        mediaId: $mediaId{j}, status: $status{j}, score: $score{j},
                        progress: $progress{j}, repeat: $repeat{j}, notes: $notes{j},
                        startedAt: $startedAt{j}, completedAt: $completedAt{j}
                    ) {{ {MediaListWithMedia.model_dump_graphql()} }}
                """)
                for k, v in json.loads(e.model_dump_json(exclude_none=True)).items():
                    variables[f"{k}{j}"] = v

            query = f"""
            mutation BatchUpdateEntries({", ".join(var_decls)}) {{
                {chr(10).join(mutations)}
            }}
            """

            try:
                response: dict[str, dict[str, dict]] = await self._make_request(
                    query, json.dumps(variables)
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
                self._list_cache[mid] = self._to_media(MediaListWithMedia(**data))
                batch_ok.add(mid)

            updated.update(batch_ok)
            missing = [by_id[mid] for mid in set(by_id) - batch_ok]
            if missing:
                updated.update(await self._fallback_update(missing))

        self._media_cache.clear()
        self._invalidate_cached_views()
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
            f"Searching for {kind} with title $$'{search_str}'$$ that is "
            f"releasing and has {episodes or 'unknown'} episodes"
        )
        for m in await self._search_anime(search_str, is_movie, limit):
            if (
                m.status == MediaStatus.RELEASING
                or m.episodes == episodes
                or not episodes
            ):
                yield m

    @ttl_cache(ttl=300)
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
        return [Media(**m) for m in response["data"]["Page"]["media"]]

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
                f"Pulling AniList data from local cache in "
                f"batched mode $${{anilist_ids: {cached_ids}}}$$"
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
            f"Pulling AniList data from API $${{anilist_id: {anilist_id}}}$$"
        )
        response = await self._make_request(query, {"id": anilist_id})
        result = Media(**response["data"]["Media"])
        self._remember(result)
        return result

    async def _batch_fetch_anime(self, anilist_ids: list[int]) -> dict[int, Media]:
        """Fetch media from AniList in batches for IDs missing in local list cache."""
        BATCH_SIZE = 50
        fetched: dict[int, Media] = {}

        for i in range(0, len(anilist_ids), BATCH_SIZE):
            batch = anilist_ids[i : i + BATCH_SIZE]
            self.log.debug(
                f"Pulling AniList data from API in batched "
                f"mode $${{anilist_ids: {batch}}}$$"
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
            response = await self._make_request(query, {"ids": batch})
            for raw in response.get("data", {}).get("Page", {}).get("media", []) or []:
                media = Media(**raw)
                fetched[media.id] = media
                self._remember(media)

        return fetched

    @ttl_cache(ttl=3600)
    async def _fetch_list_collection(self) -> MediaListCollectionWithMedia:
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

        data = MediaListCollectionWithMedia(user=self.user, has_next_chunk=True)
        refreshed_list_cache: dict[int, Media] = {}
        variables: dict[str, Any] = {
            "userId": self.user.id,
            "type": "ANIME",
            "chunk": 0,
        }

        while data.has_next_chunk:
            response = await self._make_request(query, variables)
            if refresh_epoch != self._cache_epoch:
                raise asyncio.CancelledError
            chunk = MediaListCollectionWithMedia(
                **response["data"]["MediaListCollection"]
            )
            data.has_next_chunk = chunk.has_next_chunk
            variables["chunk"] += 1

            for li in chunk.lists:
                if li.is_custom_list:
                    continue
                data.lists.append(li)
                for entry in li.entries:
                    refreshed_list_cache[entry.media_id] = self._to_media(entry)

        if refresh_epoch != self._cache_epoch:
            raise asyncio.CancelledError

        self._list_cache.clear()
        self._list_cache.update(refreshed_list_cache)
        with contextlib.suppress(AttributeError):
            self._search_anime.cache_clear()

        return data

    async def backup_anilist(self) -> str:
        """Creates a JSON backup of the user's AniList data."""
        if not self.user:
            raise aiohttp.ClientError("User information is required for deletions")

        data = await self._fetch_list_collection()

        sanitized = [
            MediaListGroup(
                entries=[
                    MediaList(
                        **{
                            f: getattr(e, f)
                            for f in MediaList.model_fields
                            if hasattr(e, f)
                        }
                    )
                    for e in li.entries
                ],
                name=li.name,
                is_custom_list=li.is_custom_list,
                is_split_completed_list=li.is_split_completed_list,
                status=li.status,
            )
            for li in data.lists
        ]

        return MediaListCollection(
            user=data.user, lists=sanitized, has_next_chunk=data.has_next_chunk
        ).model_dump_json()

    async def restore_anilist(self, backup: str) -> None:
        """Restores the user's AniList data from a JSON backup."""
        json_data = json.loads(backup)
        data = MediaListCollection(**json_data)
        await self.batch_update_anime_entries(
            [entry for li in data.lists for entry in li.entries]
        )

    @staticmethod
    def _to_media(entry: MediaListWithMedia) -> Media:
        """Converts a MediaListWithMedia object to a Media object."""
        return Media(
            media_list_entry=MediaList(
                **{
                    field: getattr(entry, field)
                    for field in MediaList.model_fields
                    if hasattr(entry, field)
                }
            ),
            **{
                field: getattr(entry.media, field)
                for field in Media.model_fields
                if hasattr(entry.media, field)
            },
        )

    @anilist_limiter
    async def _make_request(
        self, query: str, variables: dict | str | None = None, retry_count: int = 0
    ) -> dict:
        """Makes a rate-limited request to the AniList GraphQL API.

        Handles rate limiting, authentication, and automatic retries for
        rate limit exceeded responses.
        """
        if retry_count >= 3:
            raise aiohttp.ClientError("Failed to make request after 3 tries")

        session = await self._get_session()

        try:
            async with session.post(
                self.API_URL, json={"query": query, "variables": variables or {}}
            ) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    self.log.warning(
                        f"Rate limit exceeded, waiting {retry_after} seconds"
                    )
                    await asyncio.sleep(retry_after + 1)
                    return await self._make_request(query, variables, retry_count + 1)
                elif response.status == 502:
                    self.log.warning("Received 502 Bad Gateway, retrying")
                    await asyncio.sleep(1)
                    return await self._make_request(query, variables, retry_count + 1)

                try:
                    response.raise_for_status()
                except aiohttp.ClientResponseError as e:
                    self.log.error(f"Failed to make request to AniList API: {e}")
                    self.log.error(f"\t{await response.text()}")
                    self.log.error(f"\tQuery: {' '.join(query.split())}")
                    self.log.debug(f"\tVariables: {variables}")
                    raise

                return await response.json()

        except TimeoutError, aiohttp.ClientError:
            self.log.exception("Connection error while making request to AniList API")
            await asyncio.sleep(1)
            return await self._make_request(query, variables, retry_count + 1)
