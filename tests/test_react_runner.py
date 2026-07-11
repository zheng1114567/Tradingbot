"""Regression tests for the ReAct bridge import path."""
from __future__ import annotations

import inspect

from advanced_trading_agent.agents import react_runner


def test_react_runner_uses_langchain_create_agent() -> None:
    source = inspect.getsource(react_runner)

    assert "from langchain.agents import create_agent" in source
    assert "create_react_agent" not in source
