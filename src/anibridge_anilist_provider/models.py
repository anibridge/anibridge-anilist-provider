"""AniList Models Module."""

from collections.abc import Iterable
from datetime import UTC, date, datetime
from enum import StrEnum
from functools import cache
from typing import Annotated, Any, ClassVar, get_args, get_origin

from pydantic import AfterValidator, BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

UTCDateTime = Annotated[datetime, AfterValidator(lambda dt: dt.astimezone(UTC))]


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

    _ignore_ = ["__priority"]  # noqa: RUF012

    CURRENT = "CURRENT"
    PLANNING = "PLANNING"
    COMPLETED = "COMPLETED"
    DROPPED = "DROPPED"
    PAUSED = "PAUSED"
    REPEATING = "REPEATING"

    __priority: ClassVar[dict[str, int]] = {
        "PLANNING": 1,
        "CURRENT": 2,
        "PAUSED": 2,
        "DROPPED": 2,
        "COMPLETED": 3,
        "REPEATING": 3,
    }

    def __eq__(self, other: object) -> bool:
        """Check equality with another MediaListStatus."""
        if not isinstance(other, MediaListStatus):
            return NotImplemented
        return self.value == other.value

    def __ne__(self, other: object) -> bool:
        """Check inequality with another MediaListStatus."""
        if not isinstance(other, MediaListStatus):
            return NotImplemented
        return self.value != other.value

    def __lt__(self, other: object) -> bool:
        """Check if status is less than another based on priority."""
        if not isinstance(other, MediaListStatus):
            return NotImplemented
        return (
            self.value != other.value
            and self.__priority[self.value] < self.__priority[other.value]
        )

    def __le__(self, other: object) -> bool:
        """Check if status is less than or equal to another based on priority."""
        if not isinstance(other, MediaListStatus):
            return NotImplemented
        return self.__priority[self.value] <= self.__priority[other.value]

    def __gt__(self, other: object) -> bool:
        """Check if status is greater than another based on priority."""
        if not isinstance(other, MediaListStatus):
            return NotImplemented
        return (
            self.value != other.value
            and self.__priority[self.value] > self.__priority[other.value]
        )

    def __ge__(self, other: object) -> bool:
        """Check if status is greater than or equal to another based on priority."""
        if not isinstance(other, MediaListStatus):
            return NotImplemented
        return self.__priority[self.value] >= self.__priority[other.value]


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


class AnilistBaseModel(BaseModel):
    """Base, abstract class for all AniList models to represent GraphQL objects.

    Provides serialization, aliasing, and GraphQL query generation utilities.
    """

    _processed_models: ClassVar[set] = set()

    def model_dump(self, **kwargs) -> dict:
        """Convert the model to a dictionary, converting all keys to camelCase.

        Returns:
            dict: Dictionary representation of the model.
        """
        return super().model_dump(by_alias=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        """Serialize the model to JSON, converting all keys to camelCase.

        Returns:
            str: JSON serialized string of the model.
        """
        return super().model_dump_json(by_alias=True, **kwargs)

    def unset_fields(self, fields: Iterable[str]) -> None:
        """Unset specified fields to their default values.

        Args:
            fields (Iterable[str]): Field names to unset.
        """
        for field, field_info in self.__class__.model_fields.items():
            if field in fields:
                setattr(self, field, field_info.default)

    @classmethod
    @cache
    def model_dump_graphql(cls) -> str:
        """Generate GraphQL query fields for this model.

        Returns:
            str: The GraphQL query fields.
        """
        if cls.__name__ in cls._processed_models:
            return ""

        cls._processed_models.add(cls.__name__)
        fields = cls.model_fields
        graphql_fields = []

        for field_name, field in fields.items():
            field_type = (
                get_args(field.annotation)[0]
                if get_origin(field.annotation)
                else field.annotation
            )

            camel_field_name = to_camel(field_name)

            if isinstance(field_type, type) and issubclass(
                field_type, AnilistBaseModel
            ):
                nested_fields = field_type.model_dump_graphql()
                if nested_fields:
                    graphql_fields.append(f"{camel_field_name} {{\n{nested_fields}\n}}")
            else:
                graphql_fields.append(f"{camel_field_name}")

        cls._processed_models.remove(cls.__name__)
        return "\n".join(graphql_fields)

    def __hash__(self) -> int:
        """Return hash of the model representation."""
        return hash(self.__repr__())

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return f"<{
            ' : '.join(
                [f'{k}={v}' for k, v in self.model_dump().items() if v is not None]
            )
        }>"

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


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
        """Return a list of all the available titles.

        Returns:
            list[str]: All the available titles.
        """
        return [getattr(self, t) for t in self.__class__.model_fields if t]

    def __str__(self) -> str:
        """Return the first available title or an empty string.

        Returns:
            str: A title or an empty string.
        """
        return self.user_preferred or self.english or self.romaji or self.native or ""


class FuzzyDate(AnilistBaseModel):
    """Model representing a fuzzy date (year, month, day may be missing)."""

    year: int | None = None
    month: int | None = None
    day: int | None = None

    @staticmethod
    def from_date(d: date | datetime | None) -> FuzzyDate | None:
        """Create a FuzzyDate from a date or datetime object.

        Args:
            d (date | datetime | None): A date or datetime object.

        Returns:
            FuzzyDate | None: An equivalent FuzzyDate object or None.
        """
        if d is None:
            return None
        return FuzzyDate(year=d.year, month=d.month, day=d.day)

    def to_datetime(self) -> datetime | None:
        """Convert the FuzzyDate to a datetime object.

        Returns:
            datetime | None: A datetime object or None if the FuzzyDate is incomplete.
        """
        if not self.year:
            return None
        return datetime(year=self.year, month=self.month or 1, day=self.day or 1)

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
    created_at: UTCDateTime | None = None
    updated_at: UTCDateTime | None = None


class MediaListGroup[EntryType: MediaList](AnilistBaseModel):
    """Model representing a group of media list entries."""

    entries: list[EntryType] = []
    name: str | None = None
    is_custom_list: bool | None = None
    is_split_completed_list: bool | None = None
    status: MediaListStatus | None = None


class MediaListCollection[GroupType: MediaListGroup](AnilistBaseModel):
    """Model representing a collection of media list groups for a user."""

    user: User | None = None
    lists: list[GroupType] = []
    has_next_chunk: bool | None = None


class MediaCoverImage(AnilistBaseModel):
    """Model representing a media cover image."""

    extra_large: str | None = None
    large: str | None = None
    medium: str | None = None
    color: str | None = None


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


class Media(MediaWithoutList):
    """Model representing a media entry with list information."""

    media_list_entry: MediaList | None = None


class MediaListWithMedia(MediaList):
    """Model representing a media list entry with attached media info."""

    media: MediaWithoutList | None = None


class MediaListGroupWithMedia(MediaListGroup[MediaListWithMedia]):
    """Model representing a group of media list entries with media info."""

    pass


class MediaListCollectionWithMedia(MediaListCollection[MediaListGroupWithMedia]):
    """Model representing a collection of media list groups with media info."""

    pass
