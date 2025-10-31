"""Compatibility adapter for session-scoped vector store creation.

This module provides a small wrapper that delegates to :class:`VectorManager`.
Prefer importing and using VectorManager directly from `src.vector_manager`.
"""

from __future__ import annotations

from typing import Any

from .vector_manager import VectorManager


def create_session_only_vector_store(session_manager: Any, vector_manager: VectorManager):
    """Create or load a vector store for the provided session via VectorManager.

    This is a thin compatibility wrapper retained for scripts that import
    `create_session_only_vector_store` from `src.vector_session`.
    """
    return vector_manager.create_session_only_vector_store(session_manager)
