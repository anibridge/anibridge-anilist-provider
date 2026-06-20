"""AniList provider configuration."""

from typing import Annotated

import msgspec


class AnilistProviderConfig(msgspec.Struct, kw_only=True):
    """Configuration for the AniList provider."""

    token: Annotated[
        str,
        msgspec.Meta(description="AniList API token for authentication."),
    ]
    rate_limit: (
        Annotated[
            int,
            msgspec.Meta(
                ge=1,
                description=(
                    "Maximum number of API requests per minute. "
                    "Use null to rely on the shared global default limit."
                ),
            ),
        ]
        | None
    ) = None
