from __future__ import annotations

import asyncio

from knowledge_assistant.persistence import checkpoints


def test_checkpoint_loop_factory_uses_selector_loop_on_windows() -> None:
    loop = checkpoints._build_checkpoint_loop("win32")

    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()
