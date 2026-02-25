"""AniList Client."""

import asyncio
import contextlib
import importlib.metadata
import json
from collections.abc import AsyncIterator
from datetime import UTC, timedelta, timezone, tzinfo
from typing import Any

import aiohttp
from anibridge.list import ProviderLogger
from async_lru import alru_cache

from anibridge_anilist_provider.models import (
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

# The rate limit for the AniList API *should* be 90 requests per minute, but in practice
# it seems to be around 30 requests per minute


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
        self.offline_anilist_entries: dict[int, Media] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session.

        Returns:
            aiohttp.ClientSession: The active session for making HTTP requests.
        """
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
        if self._session and not self._session.closed:
            await self._session.close()

    async def initialize(self):
        """Initialize the client by getting user info and backing up data."""
        self.offline_anilist_entries.clear()
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

    async def get_user(self) -> User:
        """Retrieves the authenticated user's information from AniList.

        Makes a GraphQL query to fetch detailed user information including ID, name,
        and other profile data for the authenticated user.

        Returns:
            User: Object containing the authenticated user's information

        Raises:
            aiohttp.ClientError: If the API request fails
        """
        query = f"""
        query {{
            Viewer {{
                {User.model_dump_graphql()}
            }}
        }}
        """

        response = await self._make_request(query)
        return User(**response["data"]["Viewer"])

    async def update_anime_entry(self, media_list_entry: MediaList) -> None:
        """Updates an anime entry on the authenticated user's list.

        Sends a mutation to modify an existing anime entry in the user's list with new
        values for status, score, progress, etc.

        Args:
            media_list_entry (MediaList): Updated AniList entry to save

        Raises:
            aiohttp.ClientError: If the API request fails
        """
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

        variables = media_list_entry.model_dump_json(exclude_none=True)

        response = await self._make_request(query, variables)
        save_response = response["data"]["SaveMediaListEntry"]

        self.offline_anilist_entries[media_list_entry.media_id] = (
            self._media_list_entry_to_media(MediaListWithMedia(**save_response))
        )

    async def batch_update_anime_entries(
        self, media_list_entries: list[MediaList]
    ) -> set[int]:
        """Updates multiple anime entries on the authenticated user's list.

        Sends a batch mutation to modify multiple existing anime entries in the user's
        list. Processes entries in batches of 10 to avoid overwhelming the API.

        Args:
            media_list_entries (list[MediaList]): List of updated AniList entries to
                save.

        Returns:
            set[int]: The set of media IDs that were successfully updated.

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        BATCH_SIZE = 10

        if not media_list_entries:
            return set()

        updated_media_ids: set[int] = set()

        for i in range(0, len(media_list_entries), BATCH_SIZE):
            batch = media_list_entries[i : i + BATCH_SIZE]
            self.log.debug(
                f"Updating batch of anime entries "
                f"$${{anilist_id: {[m.media_id for m in batch]}}}$$"
            )

            entries_by_media_id = {entry.media_id: entry for entry in batch}

            variable_declarations = []
            mutation_fields = []
            variables = {}

            for j, media_list_entry in enumerate(batch):
                variable_declarations.extend(
                    [
                        f"$mediaId{j}: Int",
                        f"$status{j}: MediaListStatus",
                        f"$score{j}: Float",
                        f"$progress{j}: Int",
                        f"$repeat{j}: Int",
                        f"$notes{j}: String",
                        f"$startedAt{j}: FuzzyDateInput",
                        f"$completedAt{j}: FuzzyDateInput",
                    ]
                )
                mutation_field = f"""
                    m{j}: SaveMediaListEntry(
                        mediaId: $mediaId{j},
                        status: $status{j},
                        score: $score{j},
                        progress: $progress{j},
                        repeat: $repeat{j},
                        notes: $notes{j},
                        startedAt: $startedAt{j},
                        completedAt: $completedAt{j}
                    ) {{
                        {MediaListWithMedia.model_dump_graphql()}
                    }}
                """
                mutation_fields.append(mutation_field)

                entry_vars: dict = json.loads(
                    media_list_entry.model_dump_json(exclude_none=True)
                )
                for k, v in entry_vars.items():
                    variables[f"{k}{j}"] = v

            nl = "\n"
            query = f"""
            mutation BatchUpdateEntries({", ".join(variable_declarations)}) {{
                {nl.join(mutation_fields)}
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
                for entry in batch:
                    try:
                        await self.update_anime_entry(entry)
                        updated_media_ids.add(entry.media_id)
                    except aiohttp.ClientError as entry_exc:
                        self.log.warning(
                            "Failed to update AniList entry %s",
                            entry.media_id,
                            exc_info=entry_exc,
                        )
                continue

            updated_in_batch: set[int] = set()
            for mutation_data in response.get("data", {}).values():
                if not mutation_data or "mediaId" not in mutation_data:
                    continue
                media_id = mutation_data["mediaId"]
                self.offline_anilist_entries[media_id] = (
                    self._media_list_entry_to_media(MediaListWithMedia(**mutation_data))
                )
                updated_in_batch.add(media_id)

            updated_media_ids.update(updated_in_batch)
            missing_ids = set(entries_by_media_id) - updated_in_batch
            for media_id in missing_ids:
                entry = entries_by_media_id[media_id]
                try:
                    await self.update_anime_entry(entry)
                    updated_media_ids.add(media_id)
                except aiohttp.ClientError as entry_exc:
                    self.log.warning(
                        "Failed to update AniList entry %s",
                        media_id,
                        exc_info=entry_exc,
                    )

        return updated_media_ids

    async def delete_anime_entry(self, entry_id: int, media_id: int) -> bool:
        """Deletes an anime entry from the authenticated user's list.

        Sends a mutation to remove a specific anime entry from the user's list.

        Args:
            entry_id (int): The AniList ID of the list entry to delete.
            media_id (int): The AniList ID of the anime being deleted.

        Returns:
            bool: True if the entry was successfully deleted and not in dry run mode,
                  False otherwise.

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
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
        delete_response = response["data"]["DeleteMediaListEntry"]

        with contextlib.suppress(KeyError):
            del self.offline_anilist_entries[media_id]

        return delete_response["deleted"]

    async def search_anime(
        self,
        search_str: str,
        is_movie: bool | None,
        episodes: int | None = None,
        limit: int = 10,
    ) -> AsyncIterator[Media]:
        """Search for anime on AniList with filtering capabilities.

        Performs a search query and filters results based on media format and episode
        count. Uses local caching through _search_anime() to optimize repeated searches.

        Args:
            search_str (str): Title or keywords to search for.
            is_movie (bool | None):
                - True: search only movies and specials
                - False: search TV series, OVAs, ONAs (and TV_SHORT)
                - None: search across both movies/specials and TV/OVAs/ONAs
            episodes (int | None): Filter results to match this episode count. If None,
                returns all results.
            limit (int): Maximum number of results to return.

        Yields:
            Media: Filtered matching anime entries, sorted by relevance.

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        kind = "all" if is_movie is None else ("movie" if is_movie else "show")
        self.log.debug(
            f"Searching for {kind} "
            f"with title $$'{search_str}'$$ that is releasing and has "
            f"{episodes or 'unknown'} episodes"
        )

        res = await self._search_anime(search_str, is_movie, limit)
        for m in res:
            if (
                m.status == MediaStatus.RELEASING
                or m.episodes == episodes
                or not episodes
            ):
                yield m

    @alru_cache(ttl=300)
    async def _search_anime(
        self, search_str: str, is_movie: bool | None, limit: int = 10
    ) -> list[Media]:
        """Cached helper function for anime searches."""
        query = f"""
        query ($search: String, $formats: [MediaFormat], $limit: Int) {{
            Page(perPage: $limit) {{
                media(search: $search, type: ANIME, format_in: $formats) {{
                    {Media.model_dump_graphql()}
                }}
            }}
        }}
        """

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

        variables = {
            "search": search_str,
            "formats": formats,
            "limit": limit,
        }

        response = await self._make_request(query, variables)
        return [Media(**m) for m in response["data"]["Page"]["media"]]

    async def get_anime(self, anilist_id: int) -> Media:
        """Retrieves detailed information about a specific anime.

        Attempts to fetch anime data from local cache first, falling back to
        an API request if not found in cache.

        Args:
            anilist_id (int): The AniList ID of the anime to retrieve.

        Returns:
            Media: Detailed information about the requested anime.

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        if anilist_id in self.offline_anilist_entries:
            self.log.debug(
                f"Pulling AniList data from local cache "
                f"$${{anilist_id: {anilist_id}}}$$"
            )
            return self.offline_anilist_entries[anilist_id]

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

        self.offline_anilist_entries[anilist_id] = result

        return result

    async def batch_get_anime(self, anilist_ids: list[int]) -> list[Media]:
        """Retrieves detailed information about a list of anime.

        Attempts to fetch anime data from local cache first, falling back to
        batch API requests for entries not found in cache. Processes requests
        in batches of 10 to avoid overwhelming the API.

        Args:
            anilist_ids (list[int]): The AniList IDs of the anime to retrieve.

        Returns:
            list[Media]: Detailed information about the requested anime.

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        BATCH_SIZE = 50

        if not anilist_ids:
            return []

        result: list[Media] = []
        missing_ids = []

        cached_ids = [id for id in anilist_ids if id in self.offline_anilist_entries]
        if cached_ids:
            self.log.debug(
                f"Pulling AniList data from local cache in "
                f"batched mode $${{anilist_ids: {cached_ids}}}$$"
            )
            result.extend(self.offline_anilist_entries[id] for id in cached_ids)

        missing_ids = [
            id for id in anilist_ids if id not in self.offline_anilist_entries
        ]
        if not missing_ids:
            return result

        for i in range(0, len(missing_ids), BATCH_SIZE):
            batch_ids = missing_ids[i : i + BATCH_SIZE]
            self.log.debug(
                f"Pulling AniList data from API in batched "
                f"mode $${{anilist_ids: {batch_ids}}}$$"
            )

            query = f"""
            query BatchGetAnime($ids: [Int]) {{
                Page(perPage: {len(batch_ids)}) {{
                    media(id_in: $ids, type: ANIME) {{
                        {Media.model_dump_graphql()}
                    }}
                }}
            }}
            """

            variables = {"ids": batch_ids}
            response = await self._make_request(query, variables)

            media_list = response.get("data", {}).get("Page", {}).get("media", []) or []
            media_by_id = {m["id"]: Media(**m) for m in media_list}

            for anilist_id in batch_ids:
                media = media_by_id.get(anilist_id)
                if not media:
                    continue
                self.offline_anilist_entries[anilist_id] = media
                result.append(media)

        return result

    async def backup_anilist(self) -> str:
        """Creates a JSON backup of the user's AniList data.

        Fetches all anime entries from the user's lists and saves them to a JSON
        file. Implements a rotating backup system that maintains backups for the
        configured retention period.

        The backup includes:
            - User information
            - All non-custom anime lists
            - Detailed information about each anime entry

        Raises:
            aiohttp.ClientError: If the API request fails.
            OSError: If unable to create backup directory or write backup file.
        """
        if not self.user:
            raise aiohttp.ClientError("User information is required for deletions")

        query = f"""
        query MediaListCollection($userId: Int, $type: MediaType, $chunk: Int) {{
            MediaListCollection(userId: $userId, type: $type, chunk: $chunk) {{
                {MediaListCollectionWithMedia.model_dump_graphql()}
            }}
        }}
        """

        data = MediaListCollectionWithMedia(user=self.user, has_next_chunk=True)
        variables: dict[str, Any] = {
            "userId": self.user.id,
            "type": "ANIME",
            "chunk": 0,
        }

        while data.has_next_chunk:
            response = await self._make_request(query, variables)
            collection_data = response["data"]["MediaListCollection"]

            new_data = MediaListCollectionWithMedia(**collection_data)

            data.has_next_chunk = new_data.has_next_chunk
            variables["chunk"] += 1

            for li in new_data.lists:
                if li.is_custom_list:
                    continue
                data.lists.append(li)
                for entry in li.entries:
                    self.offline_anilist_entries[entry.media_id] = (
                        self._media_list_entry_to_media(entry)
                    )

        # To compress the backup file, remove the unecessary media field from each entry
        sanitized_lists: list[MediaListGroup] = []
        for li in data.lists:
            sanitized_entries: list[MediaList] = []
            for entry in li.entries:
                sanitized_entry = MediaList(
                    **{
                        field: getattr(entry, field)
                        for field in MediaList.model_fields
                        if hasattr(entry, field)
                    }
                )
                sanitized_entries.append(sanitized_entry)

            sanitized_lists.append(
                MediaListGroup(
                    entries=sanitized_entries,
                    name=li.name,
                    is_custom_list=li.is_custom_list,
                    is_split_completed_list=li.is_split_completed_list,
                    status=li.status,
                )
            )

        data_without_media = MediaListCollection(
            user=data.user, lists=sanitized_lists, has_next_chunk=data.has_next_chunk
        )

        return data_without_media.model_dump_json()

    async def restore_anilist(self, backup: str) -> None:
        """Restores the user's AniList data from a JSON backup.

        Parses the provided backup string and restores the user's anime lists
        and entries on AniList.

        Args:
            backup (str): The backup data as a string to be restored.
        """
        json_data = json.loads(backup)
        data = MediaListCollection(**json_data)
        await self.batch_update_anime_entries(
            [entry for li in data.lists for entry in li.entries]
        )

    def _media_list_entry_to_media(self, media_list_entry: MediaListWithMedia) -> Media:
        """Converts a MediaListWithMedia object to a Media object.

        Creates a new Media object that combines the user's list entry data
        with the anime's metadata.

        Args:
            media_list_entry (MediaListWithMedia): Combined object containing both list
                entry and media information.

        Returns:
            Media: New Media object containing all relevant fields from both the list
                entry and media information.

        Note:
            This is an internal helper method used primarily by backup_anilist().
        """
        return Media(
            media_list_entry=MediaList(
                **{
                    field: getattr(media_list_entry, field)
                    for field in MediaList.model_fields
                    if hasattr(media_list_entry, field)
                }
            ),
            **{
                field: getattr(media_list_entry.media, field)
                for field in Media.model_fields
                if hasattr(media_list_entry.media, field)
            },
        )

    async def _make_request(
        self, query: str, variables: dict | str | None = None, retry_count: int = 0
    ) -> dict:
        """Makes a rate-limited request to the AniList GraphQL API.

        Handles rate limiting, authentication, and automatic retries for
        rate limit exceeded responses.

        Args:
            query (str): GraphQL query string
            variables (dict | str): Variables for the GraphQL query
            retry_count (int): Number of retries attempted (used for temporary errors)

        Returns:
            dict: JSON response from the API

        Raises:
            aiohttp.ClientError: If the request fails for any reason other than rate
                limiting

        Note:
            - Implements rate limiting of 30 requests per minute
            - Automatically retries after waiting if rate limit is exceeded
            - Includes Authorization header using the stored token
        """
        if retry_count >= 3:
            raise aiohttp.ClientError("Failed to make request after 3 tries")

        if variables is None:
            variables = {}

        session = await self._get_session()

        try:
            async with session.post(
                self.API_URL, json={"query": query, "variables": variables}
            ) as response:
                if response.status == 429:  # Handle rate limit retries
                    retry_after = int(response.headers.get("Retry-After", 60))
                    self.log.warning(
                        f"Rate limit exceeded, waiting {retry_after} seconds"
                    )
                    await asyncio.sleep(retry_after + 1)
                    return await self._make_request(
                        query=query, variables=variables, retry_count=retry_count + 1
                    )
                elif response.status == 502:  # Bad Gateway
                    self.log.warning("Received 502 Bad Gateway, retrying")
                    await asyncio.sleep(1)
                    return await self._make_request(
                        query=query, variables=variables, retry_count=retry_count + 1
                    )

                try:
                    response.raise_for_status()
                except aiohttp.ClientResponseError as e:
                    self.log.exception("Failed to make request to AniList API")
                    response_text = await response.text()
                    self.log.debug(f"\t\t{response_text}")
                    raise e

                return await response.json()

        except TimeoutError, aiohttp.ClientError:
            self.log.exception("Connection error while making request to AniList API")
            await asyncio.sleep(1)
            return await self._make_request(
                query=query, variables=variables, retry_count=retry_count + 1
            )
