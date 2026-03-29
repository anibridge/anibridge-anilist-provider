"""Tests for limiter utility behavior used by the AniList provider."""

import pytest
from anibridge.utils.limiter import Limiter


def test_init_builds_expected_rate() -> None:
    """Initializer should preserve provided per-second token rate."""
    limiter = Limiter(120 / 60, capacity=2)

    assert limiter.rate == pytest.approx(2.0)
    assert limiter.capacity == 2


def test_per_minute_rejects_non_positive_values() -> None:
    """Initializer should reject non-positive rate values."""
    with pytest.raises(ValueError):
        Limiter(0, capacity=1)


def test_acquire_sync_mode_works() -> None:
    """Unified acquire should support blocking mode."""
    limiter = Limiter(60 / 60, capacity=1)

    assert limiter.acquire() is None


@pytest.mark.asyncio
async def test_acquire_async_mode_works() -> None:
    """Unified acquire should support async mode."""
    limiter = Limiter(60 / 60, capacity=1)

    await limiter.acquire()  # ty:ignore[invalid-await]
