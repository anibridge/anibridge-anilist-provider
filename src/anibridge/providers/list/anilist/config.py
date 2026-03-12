"""AniList provider configuration."""

from pydantic import BaseModel, Field


class AnilistListProviderConfig(BaseModel):
    """Configuration for the AniList list provider."""

    token: str = Field(default=..., description="AniList API token for authentication.")
    prefetch_list: bool = Field(
        default=False,
        description=(
            "Whether to prefetch the user's entire anime list on startup. "
            "This can improve sync performance at the cost of increased memory usage "
            "and longer startup time."
        ),
    )
