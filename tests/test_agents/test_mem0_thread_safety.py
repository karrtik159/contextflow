"""
Thread-safety tests for Mem0Service.get_client() singleton.

Verifies that concurrent threads calling get_client() result in
exactly one ``Memory.from_config()`` call (no double-init race).
"""

import threading
from unittest.mock import MagicMock, patch

from app.memory.mem0_service import Mem0Service


def test_mem0_singleton_under_contention():
    """Spawn 10 threads calling get_client() — from_config should be called once."""
    # Reset singleton state
    Mem0Service._instance = None

    fake_memory = MagicMock(name="FakeMemoryClient")

    with patch(
        "app.memory.mem0_service.Memory.from_config",
        return_value=fake_memory,
    ) as mock_from_config:
        barrier = threading.Barrier(10)
        results = []
        errors = []

        def _worker():
            try:
                barrier.wait(timeout=5)  # Synchronise all threads to start together
                client = Mem0Service.get_client()
                results.append(client)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

        # All threads should have received the same instance
        assert all(r is fake_memory for r in results)

        # from_config should have been called exactly once
        mock_from_config.assert_called_once()

    # Clean up
    Mem0Service._instance = None


def test_mem0_get_client_returns_same_instance():
    """Sequential calls return the exact same object."""
    Mem0Service._instance = None

    fake_memory = MagicMock(name="FakeMemoryClient")

    with patch(
        "app.memory.mem0_service.Memory.from_config",
        return_value=fake_memory,
    ):
        client_a = Mem0Service.get_client()
        client_b = Mem0Service.get_client()

        assert client_a is client_b
        assert client_a is fake_memory

    Mem0Service._instance = None
