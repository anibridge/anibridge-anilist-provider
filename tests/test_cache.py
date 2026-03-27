"""Regression tests for cache helpers."""

import asyncio

import pytest
from anibridge.utils.cache import ttl_cache


@pytest.mark.asyncio
async def test_ttl_cache_clear_discards_in_flight_result() -> None:
    """cache_clear should prevent an older in-flight result from being re-cached."""

    class Demo:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.resume = asyncio.Event()
            self.calls = 0

        @ttl_cache(ttl=60)
        async def load(self, value: int) -> int:
            self.calls += 1
            self.started.set()
            await self.resume.wait()
            return value * 10

    demo = Demo()
    first = asyncio.create_task(demo.load(2))
    await demo.started.wait()

    demo.load.cache_clear()
    demo.resume.set()

    assert await first == 20
    assert demo.load.cache_info().currsize == 0
    assert await demo.load(2) == 20
    assert demo.calls == 2
