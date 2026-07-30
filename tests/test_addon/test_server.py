"""Tests for BlenderServer — SO_KEEPALIVE and stale client cleanup."""

import os
import sys
import socket
import importlib.util
from unittest.mock import MagicMock
import pytest


def _load_server_module():
    """Load addon/server.py directly without triggering addon/__init__.py imports."""
    # Set up mock modules before loading
    mock_dispatcher = MagicMock()
    mock_thread_safety = MagicMock()
    mock_render_guard_module = MagicMock()
    mock_render_guard = MagicMock()
    mock_render_guard.is_rendering = False
    mock_render_guard_module.render_guard = mock_render_guard

    sys.modules.setdefault("addon", MagicMock())
    sys.modules["addon.dispatcher"] = mock_dispatcher
    sys.modules["addon.thread_safety"] = mock_thread_safety
    sys.modules["addon.render_guard"] = mock_render_guard_module

    server_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "addon", "server.py",
    )
    spec = importlib.util.spec_from_file_location("addon.server", server_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["addon.server"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def server_module():
    """Provide the loaded server module."""
    return _load_server_module()


class TestSOKeepalive:
    """Tests that accepted client sockets have SO_KEEPALIVE set."""

    def test_accepted_client_has_keepalive(self, server_module):
        """After accept(), client socket has SO_KEEPALIVE set to 1."""
        BlenderServer = server_module.BlenderServer
        server = BlenderServer()

        mock_client = MagicMock(spec=socket.socket)
        mock_server_socket = MagicMock()
        mock_server_socket.accept.side_effect = [
            (mock_client, ("127.0.0.1", 12345)),
            OSError("stop loop"),
        ]

        server._server_socket = mock_server_socket
        server._running = True

        # Run accept loop — it will accept one client then hit OSError to exit
        server._accept_loop()

        # Verify SO_KEEPALIVE was set on the accepted client
        mock_client.setsockopt.assert_any_call(
            socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1
        )


class TestClientCleanup:
    """Tests that stale/disconnected clients are removed from _clients list."""

    def test_client_removed_on_disconnect(self, server_module):
        """When _recv_message returns None, client is removed from _clients."""
        BlenderServer = server_module.BlenderServer
        server = BlenderServer()

        mock_client = MagicMock(spec=socket.socket)
        server._running = True

        # Pre-add client to the list
        with server._lock:
            server._clients.append(mock_client)

        # Simulate disconnect: _recv_message returns None
        server._recv_message = MagicMock(return_value=None)

        server._handle_client(mock_client)

        assert mock_client not in server._clients

    def test_client_socket_closed_on_disconnect(self, server_module):
        """When client disconnects, its socket is closed."""
        BlenderServer = server_module.BlenderServer
        server = BlenderServer()

        mock_client = MagicMock(spec=socket.socket)
        server._running = True

        with server._lock:
            server._clients.append(mock_client)

        server._recv_message = MagicMock(return_value=None)

        server._handle_client(mock_client)

        mock_client.close.assert_called()


class TestRenderSafeCommands:
    """Only profile-batch observation and cancellation bypass render busy."""

    @pytest.mark.parametrize(
        "command",
        [
            "get_look_render_batch",
            "cancel_look_render_batch",
            "get_look_render_result",
        ],
    )
    def test_profile_batch_commands_are_allowed(self, server_module, command):
        assert server_module._command_allowed_during_render(command) is True

    @pytest.mark.parametrize(
        "command",
        ["execute_code", "apply_light_plan", "upsert_look_profile", "render_image"],
    )
    def test_mutating_commands_remain_blocked(self, server_module, command):
        assert server_module._command_allowed_during_render(command) is False

    def test_safe_status_dispatch_bypasses_render_blocked_main_timer(
        self,
        server_module,
    ):
        server_module.dispatcher.reset_mock()
        server_module.thread_safety.reset_mock()
        server_module.render_guard.is_rendering = True
        server_module.dispatcher.dispatch.return_value = {
            "status": "ok",
            "result": {"status": "RUNNING"},
        }

        response = server_module._dispatch_command(
            "get_look_render_batch",
            {"batch_id": "lookbatch-test"},
        )

        assert response["result"]["status"] == "RUNNING"
        server_module.dispatcher.dispatch.assert_called_once_with(
            "get_look_render_batch",
            {"batch_id": "lookbatch-test"},
        )
        server_module.thread_safety.execute_on_main_thread.assert_not_called()
        server_module.render_guard.is_rendering = False

    def test_non_safe_command_still_returns_busy_without_dispatch(
        self,
        server_module,
    ):
        server_module.dispatcher.reset_mock()
        server_module.thread_safety.reset_mock()
        server_module.render_guard.is_rendering = True

        response = server_module._dispatch_command("execute_code", {})

        assert response["status"] == "busy"
        server_module.dispatcher.dispatch.assert_not_called()
        server_module.thread_safety.execute_on_main_thread.assert_not_called()
        server_module.render_guard.is_rendering = False
