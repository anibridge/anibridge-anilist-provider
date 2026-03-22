"""AniList provider configuration."""

from pydantic import BaseModel, Field


class AnilistListProviderConfig(BaseModel):
    """Configuration for the AniList list provider."""

    token: str = Field(default=..., description="AniList API token for authentication.")
