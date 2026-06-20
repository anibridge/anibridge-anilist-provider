"""AniList GraphQL models."""

from collections.abc import Iterable
from datetime import date, datetime
from enum import StrEnum
from functools import cache
from typing import Any, ClassVar, get_args, get_origin

import msgspec


class AnilistBaseEnum(StrEnum):
    """Base enum for AniList models."""

    pass


class MediaFormat(AnilistBaseEnum):
    """Enum representing media formats (TV, MOVIE, etc)."""

    TV = "TV"
    TV_SHORT = "TV_SHORT"
    MOVIE = "MOVIE"
    SPECIAL = "SPECIAL"
    OVA = "OVA"
    ONA = "ONA"
    MUSIC = "MUSIC"
    MANGA = "MANGA"
    NOVEL = "NOVEL"
    ONE_SHOT = "ONE_SHOT"


class MediaStatus(AnilistBaseEnum):
    """Enum representing media status (FINISHED, RELEASING, etc)."""

    FINISHED = "FINISHED"
    RELEASING = "RELEASING"
    NOT_YET_RELEASED = "NOT_YET_RELEASED"
    CANCELLED = "CANCELLED"
    HIATUS = "HIATUS"


class MediaListStatus(AnilistBaseEnum):
    """Enum representing status of a media list entry (CURRENT, COMPLETED, etc)."""

    CURRENT = "CURRENT"
    PLANNING = "PLANNING"
    COMPLETED = "COMPLETED"
    DROPPED = "DROPPED"
    PAUSED = "PAUSED"
    REPEATING = "REPEATING"


class Season(AnilistBaseEnum):
    """Enum representing the seasons of the year."""

    WINTER = "WINTER"
    SPRING = "SPRING"
    SUMMER = "SUMMER"
    FALL = "FALL"


class ScoreFormat(AnilistBaseEnum):
    """Enum representing score formats for media list entries."""

    POINT_100 = "POINT_100"
    POINT_10_DECIMAL = "POINT_10_DECIMAL"
    POINT_10 = "POINT_10"
    POINT_5 = "POINT_5"
    POINT_3 = "POINT_3"


class UserTitleLanguage(AnilistBaseEnum):
    """Enum representing user title language preferences."""

    ROMAJI = "ROMAJI"
    ENGLISH = "ENGLISH"
    NATIVE = "NATIVE"
    ROMAJI_STYLISED = "ROMAJI_STYLISED"
    ENGLISH_STYLISED = "ENGLISH_STYLISED"
    NATIVE_STYLISED = "NATIVE_STYLISED"


class AnilistBaseModel(msgspec.Struct, rename="camel", kw_only=True):
    """Base, abstract class for all AniList models to represent GraphQL objects.

    Provides serialization, aliasing, and GraphQL query generation utilities.
    """

    _processed_models: ClassVar[set[str]] = set()

    def unset_fields(self, fields: Iterable[str]) -> None:
        """Unset specified fields to their default values."""
        field_set = set(fields)
        for field_info in msgspec.structs.fields(type(self)):
            if field_info.name not in field_set:
                continue
            if field_info.default_factory is not msgspec.NODEFAULT:
                setattr(self, field_info.name, field_info.default_factory())
            elif field_info.default is not msgspec.NODEFAULT:
                setattr(self, field_info.name, field_info.default)

    @classmethod
    @cache
    def model_dump_graphql(cls) -> str:
        """Generate GraphQL query fields for this model."""
        if cls.__name__ in cls._processed_models:
            return ""

        cls._processed_models.add(cls.__name__)
        graphql_fields = []

        for field_info in msgspec.structs.fields(cls):
            field_type = _resolve_model_type(field_info.type)
            if field_type is not None:
                nested_fields = field_type.model_dump_graphql()
                if nested_fields:
                    graphql_fields.append(
                        f"{field_info.encode_name} {{ {nested_fields} }}"
                    )
            else:
                graphql_fields.append(f"{field_info.encode_name}")

        cls._processed_models.remove(cls.__name__)
        return " ".join(graphql_fields)

    def __hash__(self) -> int:
        """Return hash of the model representation."""
        return hash(self.__repr__())

    def __repr__(self) -> str:
        """Return string representation of the model."""
        values = [
            f"{k}={v}" for k, v in msgspec.to_builtins(self).items() if v is not None
        ]
        return f"<{' : '.join(values)}>"


def _resolve_model_type(annotation: Any) -> type[AnilistBaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, AnilistBaseModel):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        return None

    for arg in get_args(annotation):
        resolved = _resolve_model_type(arg)
        if resolved is not None:
            return resolved
    return None


class MediaListOptions(AnilistBaseModel):
    """Model representing media list options for a user."""

    score_format: ScoreFormat | None = None
    row_order: str | None = None


class UserOptions(AnilistBaseModel):
    """Model representing user options/preferences."""

    title_language: UserTitleLanguage | None = None
    timezone: str | None = None


class User(AnilistBaseModel):
    """Model representing an AniList user."""

    id: int
    name: str
    media_list_options: MediaListOptions | None = None
    options: UserOptions | None = None


class MediaTitle(AnilistBaseModel):
    """Model representing media titles in various languages."""

    romaji: str | None = None
    english: str | None = None
    native: str | None = None
    user_preferred: str | None = None

    def titles(self) -> list[str]:
        """Return a list of all the available titles."""
        return [getattr(self, field) for field in self.__struct_fields__ if field]

    def best_title(self) -> str:
        """Return the preferred title using AniList's user-preferred fallback."""
        return self.user_preferred or self.romaji or self.english or self.native or ""

    def __str__(self) -> str:
        """Return the first available title or an empty string."""
        return self.user_preferred or self.english or self.romaji or self.native or ""


class FuzzyDate(AnilistBaseModel):
    """Model representing a fuzzy date (year, month, day may be missing)."""

    year: int | None = None
    month: int | None = None
    day: int | None = None

    @staticmethod
    def from_date(d: date | datetime | None) -> FuzzyDate | None:
        """Create a FuzzyDate from a date or datetime object."""
        if d is None:
            return None
        return FuzzyDate(year=d.year, month=d.month, day=d.day)

    def to_date(self) -> date | None:
        """Convert the FuzzyDate to a date object, filling defaults."""
        if self.year is None:
            return None
        month = self.month if self.month is not None else 1
        day = self.day if self.day is not None else 1
        return date(year=self.year, month=month, day=day)

    def __bool__(self) -> bool:
        """Return True if the date has a year, else False."""
        return self.year is not None

    def __eq__(self, other: Any) -> bool:
        """Check equality with another FuzzyDate."""
        if not isinstance(other, FuzzyDate):
            return False
        return (
            self.year == other.year
            and self.month == other.month
            and self.day == other.day
        )

    def __lt__(self, other: Any) -> bool:
        """Check if this date is less than another."""
        if not isinstance(other, FuzzyDate):
            return NotImplemented
        if not self.year or not other.year:
            return True
        return ((self.year), (self.month or 1), (self.day or 1)) < (
            (other.year),
            (other.month or 1),
            (other.day or 1),
        )

    def __le__(self, other: Any) -> bool:
        """Check if this date is less than or equal to another."""
        if not isinstance(other, FuzzyDate):
            return NotImplemented
        if not self.year or not other.year:
            return True
        return ((self.year), (self.month or 1), (self.day or 1)) <= (
            (other.year),
            (other.month or 1),
            (other.day or 1),
        )

    def __gt__(self, other: Any) -> bool:
        """Check if this date is greater than another."""
        if not isinstance(other, FuzzyDate):
            return NotImplemented
        if not self.year or not other.year:
            return True
        return ((self.year), (self.month or 1), (self.day or 1)) > (
            (other.year),
            (other.month or 1),
            (other.day or 1),
        )

    def __ge__(self, other: Any) -> bool:
        """Check if this date is greater than or equal to another."""
        if not isinstance(other, FuzzyDate):
            return NotImplemented
        if not self.year or not other.year:
            return True
        return ((self.year), (self.month or 1), (self.day or 1)) >= (
            (other.year),
            (other.month or 1),
            (other.day or 1),
        )

    def __str__(self) -> str:
        """Return string representation of the FuzzyDate."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Return formatted string representation of the FuzzyDate."""
        return (
            f"{self.year or '????'}-"
            f"{str(self.month).zfill(2) if self.month else '??'}-"
            f"{str(self.day).zfill(2) if self.day else '??'}"
        )


class MediaList(AnilistBaseModel):
    """Model representing a media list entry in AniList."""

    id: int
    user_id: int
    media_id: int
    status: MediaListStatus | None = None
    score: float | None = None
    progress: int | None = None
    repeat: int | None = None
    notes: str | None = None
    started_at: FuzzyDate | None = None
    completed_at: FuzzyDate | None = None
    created_at: int | None = None
    updated_at: int | None = None


class MediaListGroup(AnilistBaseModel):
    """Model representing a group of media list entries."""

    entries: list[MediaList] = msgspec.field(default_factory=list)
    name: str | None = None
    is_custom_list: bool | None = None
    is_split_completed_list: bool | None = None
    status: MediaListStatus | None = None


class MediaListCollection(AnilistBaseModel):
    """Model representing a collection of media list groups for a user."""

    user: User | None = None
    lists: list[MediaListGroup] = msgspec.field(default_factory=list)
    has_next_chunk: bool | None = None


class MediaCoverImage(AnilistBaseModel):
    """Model representing a media cover image."""

    # extra_large: str | None = None
    # large: str | None = None
    medium: str | None = None
    # color: str | None = None


class MediaWithoutList(AnilistBaseModel):
    """Model representing a media entry without list information."""

    id: int
    # id_mal: int | None = None
    format: MediaFormat | None = None
    status: MediaStatus | None = None
    episodes: int | None = None
    cover_image: MediaCoverImage | None = None
    # banner_image: str | None = None
    # synonyms: list[str] | None = None
    # is_locked: bool | None = None
    title: MediaTitle | None = None
    # start_date: FuzzyDate | None = None
    # end_date: FuzzyDate | None = None
    # next_airing_episode: dict | None = None
    season: Season | None = None
    season_year: int | None = None


class Media(MediaWithoutList):
    """Model representing a media entry with list information."""

    media_list_entry: MediaList | None = None


class MediaListWithMedia(MediaList):
    """Model representing a media list entry with attached media info."""

    media: MediaWithoutList | None = None


class MediaListGroupWithMedia(MediaListGroup):
    """Model representing a group of media list entries with media info."""

    entries: list[MediaListWithMedia] = msgspec.field(default_factory=list)


class MediaListCollectionWithMedia(MediaListCollection):
    """Model representing a collection of media list groups with media info."""

    lists: list[MediaListGroupWithMedia] = msgspec.field(default_factory=list)
