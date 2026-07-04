"""AniBridge provider implementation for AniList."""

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, cast

import aiohttp
import msgspec
from anibridge.provider.base import (
    Account,
    Artwork,
    BackupArtifact,
    Capabilities,
    Delete,
    Descriptor,
    Event,
    ExternalId,
    FacetName,
    FieldSpec,
    Match,
    Node,
    NodeKind,
    NodeQuery,
    NodeSpec,
    NumericConstraint,
    Page,
    Progress,
    ProgressConstraint,
    Provider,
    Query,
    Rating,
    Record,
    RecordField,
    RecordQuery,
    RecordSpec,
    Ref,
    ResourceKind,
    Role,
    State,
    Status,
    SupportsBackupExports,
    SupportsMapping,
    SupportsNodeSearch,
    SupportsReads,
    SupportsWrites,
    TemporalConstraint,
    TemporalPrecision,
    TextConstraint,
    UpsertRecord,
    Value,
    Write,
    WriteAction,
    WriteError,
    WriteResult,
)

from anibridge.providers.anilist.client import AnilistClient
from anibridge.providers.anilist.config import AnilistProviderConfig
from anibridge.providers.anilist.models import (
    FuzzyDate,
    Media,
    MediaListStatus,
    MediaListWithMedia,
    ScoreFormat,
)

__all__ = ["AnilistProvider"]

_MEDIA_LIST_SURFACE = "media_list"

_STATUS_TO_NATIVE: dict[Status, MediaListStatus] = {
    Status.ACTIVE: MediaListStatus.CURRENT,
    Status.PLANNED: MediaListStatus.PLANNING,
    Status.COMPLETED: MediaListStatus.COMPLETED,
    Status.DROPPED: MediaListStatus.DROPPED,
    Status.PAUSED: MediaListStatus.PAUSED,
    Status.REPEATING: MediaListStatus.REPEATING,
}
_NATIVE_TO_STATUS: dict[MediaListStatus, Status] = {
    native: status for status, native in _STATUS_TO_NATIVE.items()
}


class AnilistProvider(
    Provider,
    SupportsMapping,
    SupportsNodeSearch,
    SupportsReads,
    SupportsWrites,
    SupportsBackupExports,
):
    """AniList target provider for the AniBridge provider contract."""

    DISPLAY_NAME = "AniList"
    NAMESPACE = "anilist"

    def __init__(
        self,
        *,
        logger,
        config: Mapping[str, object] | None = None,
    ) -> None:
        """Parse configuration and prepare the AniList client."""
        super().__init__(logger=logger, config=config)
        self.parsed_config = msgspec.convert(config or {}, type=AnilistProviderConfig)
        self._client = AnilistClient(
            anilist_token=self.parsed_config.token,
            logger=self.log,
            rate_limit=self.parsed_config.rate_limit,
        )
        self._account: Account | None = None
        self._score_format: ScoreFormat = ScoreFormat.POINT_100

    async def initialize(self) -> None:
        """Initialize the AniList API session and user cache."""
        self.log.debug("Initializing AniList provider client")
        await self._client.initialize()
        if self._client.user is None:
            raise RuntimeError("Failed to fetch AniList user during initialization")
        self._account = Account(
            key=str(self._client.user.id),
            title=self._client.user.name,
            url=f"https://anilist.co/user/{self._client.user.name}",
        )
        options = self._client.user.media_list_options
        if options is not None and options.score_format is not None:
            self._score_format = options.score_format
        self.log.debug("AniList provider initialized for user id=%s", self._account.key)

    def account(self) -> Account | None:
        """Return the connected AniList account."""
        return self._account

    def capabilities(self) -> Capabilities:
        """Advertise AniList target capabilities."""
        return Capabilities(
            roles=frozenset({Role.TARGET}),
            facets=frozenset({FacetName.ARTWORK}),
            specs=(
                NodeSpec(kind=Descriptor("anime", NodeKind.SERIES)),
                RecordSpec(
                    name=_MEDIA_LIST_SURFACE,
                    fields={
                        RecordField.STATUS: FieldSpec(
                            RecordField.STATUS,
                            readable=True,
                            writable=True,
                            values=tuple(
                                Descriptor(native.value, status)
                                for status, native in _STATUS_TO_NATIVE.items()
                            ),
                        ),
                        RecordField.PROGRESS: FieldSpec(
                            RecordField.PROGRESS,
                            readable=True,
                            writable=True,
                            constraints=(
                                ProgressConstraint(
                                    current=NumericConstraint(0, None, 1),
                                    total=False,
                                    unit=False,
                                ),
                            ),
                        ),
                        RecordField.RATING: FieldSpec(
                            RecordField.RATING,
                            readable=True,
                            writable=True,
                            constraints=(self._rating_constraint(),),
                        ),
                        RecordField.STARTED_AT: FieldSpec(
                            RecordField.STARTED_AT,
                            readable=True,
                            writable=True,
                            constraints=(
                                TemporalConstraint(precision=TemporalPrecision.DATE),
                            ),
                        ),
                        RecordField.FINISHED_AT: FieldSpec(
                            RecordField.FINISHED_AT,
                            readable=True,
                            writable=True,
                            constraints=(
                                TemporalConstraint(precision=TemporalPrecision.DATE),
                            ),
                        ),
                        RecordField.REPEAT_COUNT: FieldSpec(
                            RecordField.REPEAT_COUNT,
                            readable=True,
                            writable=True,
                            constraints=(NumericConstraint(0, None, 1),),
                        ),
                        RecordField.NOTES: FieldSpec(
                            RecordField.NOTES,
                            readable=True,
                            writable=True,
                            constraints=(TextConstraint(max_length=1000),),
                        ),
                    },
                    write_actions=frozenset({WriteAction.UPSERT, WriteAction.DELETE}),
                ),
            ),
            external_authorities=frozenset({"anilist"}),
        )

    async def close(self) -> None:
        """Close the AniList API session."""
        await self._client.close()

    async def clear_cache(self) -> None:
        """Clear AniList provider caches."""
        self._client.clear_cache()

    async def export_backup(self) -> BackupArtifact | None:
        """Export the AniList list as a provider-managed backup artifact."""
        payload = await self._client.backup_anilist()
        return BackupArtifact(
            content=payload.encode(),
            file_extension=".json",
            media_type="application/json",
        )

    async def resolve(self, ids: Sequence[ExternalId]) -> Sequence[Match]:
        """Resolve AniList external IDs to AniList refs."""
        matches: list[Match] = []
        for external_id in ids:
            if external_id.authority != self.NAMESPACE:
                continue
            try:
                int(external_id.value)
            except ValueError:
                continue
            matches.append(
                Match(
                    external_id=external_id,
                    ref=Ref.anchor(external_id.value),
                    confidence=1.0,
                )
            )
        return tuple(matches)

    async def fetch(self, query: Query) -> Page[Node | Record | Event]:
        """Fetch AniList nodes or records matching a typed query."""
        if isinstance(query, NodeQuery):
            return cast(Page[Node | Record | Event], await self._fetch_nodes(query))
        if isinstance(query, RecordQuery):
            return cast(Page[Node | Record | Event], await self._fetch_records(query))
        return Page(items=())

    async def _fetch_nodes(self, query: NodeQuery) -> Page[Node]:
        """Fetch AniList media metadata for targeted refs."""
        if query.native_kinds and "anime" not in query.native_kinds:
            return Page(items=())

        media_ids: list[int] = []
        for ref in query.refs:
            if query.limit is not None and len(media_ids) >= query.limit:
                break
            try:
                media_ids.append(int(ref.key))
            except ValueError:
                self.log.warning("Invalid AniList media ref %s", ref.key)

        media_items = await self._client.batch_get_anime(media_ids)
        return Page(
            items=tuple(
                self._node_from_media(media, query.facets) for media in media_items
            )
        )

    async def search_nodes(
        self,
        query: str,
        *,
        limit: int = 10,
        facets: frozenset[FacetName] = frozenset(),
    ) -> Page[Node]:
        """Search AniList anime by title."""
        text = query.strip()
        if not text:
            return Page(items=())
        media_items = [
            media
            async for media in self._client.search_anime(
                text,
                is_movie=None,
                limit=limit,
            )
        ]
        return Page(
            items=tuple(self._node_from_media(media, facets) for media in media_items)
        )

    async def _fetch_records(self, query: RecordQuery) -> Page[Record]:
        """Fetch AniList media-list records by ref or record key."""
        refs = tuple(query.refs)
        if not refs and query.keys:
            refs = tuple(Ref.anchor(key) for key in query.keys)
        if query.native_kinds and _MEDIA_LIST_SURFACE not in query.native_kinds:
            return Page(items=())

        records: list[Record] = []
        for ref in refs:
            if query.limit is not None and len(records) >= query.limit:
                break
            try:
                media_id = int(ref.key)
            except ValueError:
                self.log.warning("Invalid AniList media ref %s", ref.key)
                continue

            media = await self._client.get_anime(media_id)

            records.append(self._record_from_media(media, query.fields))
        return Page(items=tuple(records))

    async def write(self, writes: Sequence[Write]) -> Sequence[WriteResult]:
        """Apply AniList record writes."""
        results: list[WriteResult] = []
        for write in writes:
            try:
                if isinstance(write, UpsertRecord):
                    result = await self._upsert_record(write)
                elif (
                    isinstance(write, Delete) and write.resource is ResourceKind.RECORD
                ):
                    result = await self._delete_record(write)
                else:
                    result = WriteResult(
                        ok=False,
                        resource=write.resource,
                        action=write.action,
                        token=write.token,
                        code=WriteError.UNSUPPORTED,
                        error="AniList only supports record writes",
                    )
            except Exception as exc:
                result = WriteResult(
                    ok=False,
                    resource=write.resource,
                    action=write.action,
                    token=write.token,
                    code=self._write_error_for_exception(exc),
                    error=str(exc),
                    ref=getattr(write, "ref", None),
                )
            results.append(result)
        return tuple(results)

    def _record_from_media(
        self,
        media: Media,
        fields: frozenset[RecordField],
    ) -> Record:
        """Convert AniList media/list state into a contract record."""
        requested = fields or frozenset(RecordField)
        entry = media.media_list_entry
        values: dict[RecordField, Value] = {}
        if entry is not None:
            if RecordField.STATUS in requested and entry.status is not None:
                values[RecordField.STATUS] = self._state_from_native(entry.status)
            if RecordField.PROGRESS in requested and entry.progress is not None:
                values[RecordField.PROGRESS] = Progress(
                    current=entry.progress,
                    total=media.episodes,
                    unit="episode",
                )
            if RecordField.RATING in requested and entry.score is not None:
                values[RecordField.RATING] = self._rating_from_score(entry.score)
            if RecordField.REPEAT_COUNT in requested and entry.repeat is not None:
                values[RecordField.REPEAT_COUNT] = entry.repeat
            if RecordField.NOTES in requested and entry.notes:
                values[RecordField.NOTES] = entry.notes
            if RecordField.STARTED_AT in requested and entry.started_at is not None:
                started = entry.started_at.to_date()
                if started is not None:
                    values[RecordField.STARTED_AT] = started
            if RecordField.FINISHED_AT in requested and entry.completed_at is not None:
                completed = entry.completed_at.to_date()
                if completed is not None:
                    values[RecordField.FINISHED_AT] = completed

        return Record(
            ref=Ref.anchor(str(media.id)),
            surface=_MEDIA_LIST_SURFACE,
            key=str(entry.id) if entry is not None and entry.id else None,
            ids=(ExternalId(self.NAMESPACE, str(media.id)),),
            values=values,
            updated_at=(
                datetime.fromtimestamp(entry.updated_at, UTC)
                if entry is not None and entry.updated_at is not None
                else None
            ),
            url=f"https://anilist.co/anime/{media.id}",
        )

    def _node_from_media(
        self,
        media: Media,
        facets: frozenset[FacetName],
    ) -> Node:
        """Convert AniList media metadata into a contract node."""
        hydrated = {}
        if FacetName.ARTWORK in facets and media.cover_image is not None:
            poster = media.cover_image.medium
            if poster:
                hydrated[FacetName.ARTWORK] = Artwork({"poster": poster})

        return Node(
            ref=Ref.anchor(str(media.id)),
            kind="anime",
            title=media.title.best_title() if media.title else None,
            url=f"https://anilist.co/anime/{media.id}",
            labels=self._labels_for_media(media),
            facets=hydrated,
        )

    @staticmethod
    def _labels_for_media(media: Media) -> tuple[str, ...]:
        """Return concise AniList labels for timeline display."""
        labels: list[str] = []
        if media.season and media.season_year:
            labels.append(f"{media.season.value.title()} {media.season_year}")
        elif media.season_year:
            labels.append(str(media.season_year))
        if media.format:
            labels.append(media.format.value.replace("_", " ").title())
        if media.status:
            labels.append(media.status.value.replace("_", " ").title())
        return tuple(labels)

    async def _upsert_record(self, write: UpsertRecord) -> WriteResult:
        """Apply one upsert record write."""
        ref = write.ref
        media_id = int(ref.key)
        changed_fields = frozenset((*write.set.keys(), *write.clear))
        if not changed_fields:
            return WriteResult(
                ok=True,
                resource=ResourceKind.RECORD,
                action=WriteAction.UPSERT,
                token=write.token,
                ref=ref,
            )
        variables = self._variables_for_write(media_id, write)
        saved = await self._save_media_list_entry(variables)
        media = self._client._to_media(saved)
        self._client._list_cache[media_id] = media
        self._client._media_cache.pop(media_id, None)
        self._client._invalidate_cached_views(clear_list_collection_cache=False)
        self._client._mark_list_cache_dirty()
        return WriteResult(
            ok=True,
            resource=ResourceKind.RECORD,
            action=WriteAction.UPSERT,
            token=write.token,
            key=str(saved.id),
            ref=write.ref,
            revision=str(saved.updated_at) if saved.updated_at is not None else None,
        )

    async def _delete_record(self, write: Delete) -> WriteResult:
        """Delete one AniList record."""
        ref = write.ref
        if ref is None:
            return WriteResult(
                ok=False,
                resource=ResourceKind.RECORD,
                action=WriteAction.DELETE,
                token=write.token,
                code=WriteError.INVALID,
                error="AniList delete requires a ref",
            )
        media = await self._client.get_anime(int(ref.key))
        entry = media.media_list_entry
        if entry is None:
            return WriteResult(
                ok=True,
                resource=ResourceKind.RECORD,
                action=WriteAction.DELETE,
                token=write.token,
                ref=ref,
            )
        deleted = await self._client.delete_anime_entry(entry.id, entry.media_id)
        return WriteResult(
            ok=bool(deleted),
            resource=ResourceKind.RECORD,
            action=WriteAction.DELETE,
            token=write.token,
            ref=ref,
            key=str(entry.id),
            code=None if deleted else WriteError.INTERNAL,
            error=None if deleted else "AniList delete returned false",
        )

    def _variables_for_write(
        self,
        media_id: int,
        write: UpsertRecord,
    ) -> dict[str, Any]:
        """Translate contract set/clear fields into AniList mutation variables."""
        variables: dict[str, Any] = {"mediaId": media_id}
        for field in write.clear:
            variables[self._field_variable(field)] = None
        for field, value in write.set.items():
            variable = self._field_variable(field)
            variables[variable] = self._to_anilist_value(field, value)
        return variables

    async def _save_media_list_entry(
        self,
        variables: dict[str, Any],
    ) -> MediaListWithMedia:
        """Run a dynamic AniList SaveMediaListEntry mutation."""
        variable_types = {
            "mediaId": "Int",
            "status": "MediaListStatus",
            "progress": "Int",
            "score": "Float",
            "repeat": "Int",
            "notes": "String",
            "startedAt": "FuzzyDateInput",
            "completedAt": "FuzzyDateInput",
        }
        variable_order = [
            "mediaId",
            "status",
            "progress",
            "score",
            "repeat",
            "notes",
            "startedAt",
            "completedAt",
        ]
        present = [name for name in variable_order if name in variables]
        declarations = ", ".join(f"${name}: {variable_types[name]}" for name in present)
        args = ", ".join(f"{name}: ${name}" for name in present)
        query = f"""
        mutation SaveAniBridgeEntry({declarations}) {{
            SaveMediaListEntry({args}) {{
                {MediaListWithMedia.model_dump_graphql()}
            }}
        }}
        """
        response = await self._client._make_request(query, variables)
        return msgspec.convert(
            response["data"]["SaveMediaListEntry"],
            type=MediaListWithMedia,
        )

    @staticmethod
    def _field_variable(field: RecordField) -> str:
        """Return the AniList mutation variable for a record field."""
        match field:
            case RecordField.STATUS:
                return "status"
            case RecordField.PROGRESS:
                return "progress"
            case RecordField.RATING:
                return "score"
            case RecordField.REPEAT_COUNT:
                return "repeat"
            case RecordField.NOTES:
                return "notes"
            case RecordField.STARTED_AT:
                return "startedAt"
            case RecordField.FINISHED_AT:
                return "completedAt"
            case _:
                raise ValueError(f"AniList cannot write field {field.value!r}")

    def _to_anilist_value(self, field: RecordField, value: object) -> object:
        """Translate one contract value into an AniList mutation value."""
        match field:
            case RecordField.STATUS:
                status = value.status if isinstance(value, State) else value
                if not isinstance(status, Status):
                    return None
                return _STATUS_TO_NATIVE[status].value
            case RecordField.PROGRESS:
                if isinstance(value, Progress):
                    return int(value.current or 0)
                if not isinstance(value, int | float) or isinstance(value, bool):
                    raise ValueError("progress must be numeric")
                return int(value)
            case RecordField.RATING:
                if isinstance(value, Rating):
                    return self._score_from_rating(value)
                if not isinstance(value, int | float) or isinstance(value, bool):
                    raise ValueError("rating must be numeric")
                return float(value)
            case RecordField.REPEAT_COUNT:
                if not isinstance(value, int | float) or isinstance(value, bool):
                    raise ValueError("repeat_count must be numeric")
                return int(value)
            case RecordField.NOTES:
                return str(value)
            case RecordField.STARTED_AT | RecordField.FINISHED_AT:
                if isinstance(value, datetime):
                    if value.tzinfo is None or value.utcoffset() is None:
                        raise ValueError(f"{field.value} must be timezone-aware")
                    value = value.astimezone(UTC).date()
                if not isinstance(value, date):
                    raise ValueError(f"{field.value} must be date")
                return msgspec.to_builtins(FuzzyDate.from_date(value))
            case _:
                raise ValueError(f"AniList cannot write field {field.value!r}")

    @staticmethod
    def _state_from_native(status: MediaListStatus) -> State:
        """Convert an AniList native status into contract state."""
        return State(native=status.value, status=_NATIVE_TO_STATUS.get(status))

    def _rating_from_score(self, score: float) -> Rating:
        """Convert an AniList score into the advertised native scale."""
        minimum, maximum, step = self._rating_scale()
        return Rating(float(score), (minimum, maximum, step))

    def _score_from_rating(self, rating: Rating) -> float:
        """Convert a rating value into the current AniList score format."""
        return float(rating.value)

    def _rating_constraint(self) -> NumericConstraint:
        """Return the rating constraint for the current user's score format."""
        minimum, maximum, step = self._rating_scale()
        return NumericConstraint(minimum=minimum, maximum=maximum, step=step)

    def _rating_scale(self) -> tuple[float, float, float]:
        """Return AniList's active score scale."""
        match self._score_format:
            case ScoreFormat.POINT_10_DECIMAL:
                return (0, 10, 0.1)
            case ScoreFormat.POINT_10:
                return (0, 10, 1)
            case ScoreFormat.POINT_5:
                return (0, 5, 1)
            case ScoreFormat.POINT_3:
                return (0, 3, 1)
            case _:
                return (0, 100, 1)

    @staticmethod
    def _write_error_for_exception(exc: Exception) -> WriteError:
        """Classify an exception into a contract write error."""
        if isinstance(exc, ValueError):
            return WriteError.INVALID
        if isinstance(exc, aiohttp.ClientResponseError):
            if exc.status in {401, 403}:
                return WriteError.AUTH
            if exc.status == 404:
                return WriteError.NOT_FOUND
            if exc.status == 429:
                return WriteError.RATE_LIMITED
        if isinstance(exc, aiohttp.ClientError):
            return WriteError.TRANSIENT
        return WriteError.INTERNAL
