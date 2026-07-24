"""Integration test: gateway/platforms/webhook.py's approval-delegate wiring
registers/unregisters under the REAL gateway approval session_key.

Regression guard for Copilot review findings on TechClusiveSolutions/
hermes-agent#2 (findings 1, 2, 4): the delegate must register under
``gateway.session.build_session_key(source, ...)``'s output — not a raw
webhook ``chat_id`` — because that's the exact key ``tools.approval`` waits
on and the exact key ``BasePlatformAdapter.handle_message`` independently
computes for the same source moments later. Registering under the wrong key
means the notify_cb is silently never found. The registration must also be
pinned, or ``gateway/run.py``'s per-turn agent runner clobbers it.

These tests drive the real HTTP handler (``_handle_webhook`` via aiohttp's
TestClient/TestServer), mirroring tests/gateway/test_webhook_adapter.py's
pattern, so the actual registration call site is exercised — not a
hand-constructed MessageEvent that bypasses it.
"""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from gateway.session import SessionSource, build_session_key


def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application(client_max_size=adapter._max_body_bytes)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _expected_session_key(adapter: WebhookAdapter, route_name: str, delivery_id: str) -> str:
    """What BasePlatformAdapter.handle_message will independently compute
    for this delivery's source — the key the delegate MUST register under.
    ``delivery_id`` must be the value actually used by ``_handle_webhook``:
    pass it explicitly via the ``X-GitHub-Delivery`` header so it's known
    rather than guessed (the adapter auto-generates one otherwise)."""
    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id=f"webhook:{route_name}:{delivery_id}",
        chat_name=f"webhook/{route_name}",
        chat_type="webhook",
        user_id=f"webhook:{route_name}",
        user_name=route_name,
    )
    return build_session_key(
        source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )


class _FakeButtonAdapter:
    typed_command_prefix = "/"

    async def send_exec_approval(self, **kwargs):
        return SendResult(success=True)

    async def send(self, chat_id, content, metadata=None):
        return SendResult(success=True)


class _FakeRunner:
    def __init__(self, adapters):
        self.adapters = adapters
        self._profile_adapters = {}

    def _read_user_config(self):
        return {}

    def _profile_name_for_source(self, *args, **kwargs):
        return None


async def _drain_background_tasks(adapter: WebhookAdapter, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while adapter._background_tasks and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_handle_webhook_registers_delegate_under_the_real_session_key():
    """The delegate must be registered under the SAME key
    BasePlatformAdapter.handle_message computes for this delivery's source —
    not the raw session_chat_id.

    Captures the actual session_key ``maybe_register_delegate`` is called
    with (via a wrapping patch that still runs the real implementation)
    rather than inspecting tools.approval's live registry after the fact —
    the stubbed _message_handler returns instantly, so the background
    agent-run task can race ahead and unregister before a post-hoc registry
    check would run, which isn't the thing under test here.
    """
    import gateway.approval_delegate as delegate_mod

    adapter = _make_adapter(
        {
            "alerts": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Alert: {message}",
                "deliver": "log",
                "approval_delegate": "slack:C0B8JK868SX",
            }
        }
    )
    adapter.gateway_runner = _FakeRunner(adapters={Platform.SLACK: _FakeButtonAdapter()})

    async def _message_handler(event: MessageEvent):
        return ""

    adapter._message_handler = _message_handler

    from tools import approval as mod

    real_maybe_register_delegate = delegate_mod.maybe_register_delegate
    calls = []

    def _spy(session_key, *args, **kwargs):
        calls.append(session_key)
        return real_maybe_register_delegate(session_key, *args, **kwargs)

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(delegate_mod, "maybe_register_delegate", _spy)
            resp = await cli.post(
                "/webhooks/alerts",
                json={"message": "server on fire"},
                headers={"X-GitHub-Delivery": "delivery-key-test-001"},
            )
            assert resp.status == 202
            await _drain_background_tasks(adapter)

    assert len(calls) == 1, f"expected exactly one registration attempt, got {calls}"
    expected_key = _expected_session_key(adapter, "alerts", "delivery-key-test-001")
    assert calls[0] == expected_key, (
        f"delegate registered under {calls[0]!r}, but the real gateway approval "
        f"session_key is {expected_key!r} — tools.approval will never find it there"
    )
    # Session ended and drained — must not have leaked past it.
    assert expected_key not in mod._gateway_notify_cbs
    assert expected_key not in mod._gateway_notify_pinned


@pytest.mark.asyncio
async def test_on_processing_complete_unregisters_the_same_real_session_key():
    """Unregistration must target the identical key registration used, or the
    notify_cb (and its pin) leaks past the one-shot webhook session's end."""
    adapter = _make_adapter(
        {
            "alerts": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Alert: {message}",
                "deliver": "log",
                "approval_delegate": "slack:C0B8JK868SX",
            }
        }
    )
    adapter.gateway_runner = _FakeRunner(adapters={Platform.SLACK: _FakeButtonAdapter()})

    async def _message_handler(event: MessageEvent):
        return ""

    adapter._message_handler = _message_handler

    from tools import approval as mod

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/webhooks/alerts",
            json={"message": "server on fire"},
            headers={"X-GitHub-Delivery": "delivery-key-test-002"},
        )
        assert resp.status == 202

        await _drain_background_tasks(adapter)

        expected_key = _expected_session_key(adapter, "alerts", "delivery-key-test-002")
        assert expected_key not in mod._gateway_notify_cbs, (
            "notify_cb leaked past webhook session end — on_processing_complete "
            "must unregister under the same key used at registration"
        )
        assert expected_key not in mod._gateway_notify_pinned
