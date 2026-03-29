"""AniList provider configuration."""

from pydantic import BaseModel, Field


class AnilistListProviderConfig(BaseModel):
    """Configuration for the AniList list provider."""

    token: str = Field(default=..., description="AniList API token for authentication.")
    rate_limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of API requests per minute. "
            "Use null to rely on the shared global default limit."
        ),
    )
