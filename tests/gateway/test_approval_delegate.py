"""Unit tests for gateway/approval_delegate.py.

Covers the acceptance criteria in TechClusiveSolutions/hermes-agent#1:
no-op when unconfigured, correct target parsing, adapter-not-connected /
malformed-target fail-closed paths, button-based send with text fallback,
and the session-scoped approval-timeout override.

See hermes-webhook-approval-delegate-design.md (Part 3a) for the design
this implements.
"""

import asyncio

import pytest

from gateway.approval_delegate import (
    _get_delegate_adapter,
    _parse_delegate_target,
    _resolve_delegate_target,
    _resolve_delegate_timeout,
    maybe_register_delegate,
)
from gateway.config import Platform
from gateway.platforms.base import SendResult


class _FakeButtonAdapter:
    """Adapter stub that supports send_exec_approval (button path)."""

    typed_command_prefix = "/"

    def __init__(self, *, approval_success=True):
        self._approval_success = approval_success
        self.approval_calls = []
        self.text_calls = []

    async def send_exec_approval(self, **kwargs):
        self.approval_calls.append(kwargs)
        return SendResult(
            success=self._approval_success,
            error=None if self._approval_success else "adapter rejected send",
        )

    async def send(self, chat_id, content, metadata=None):
        self.text_calls.append((chat_id, content, metadata))
        return SendResult(success=True)


class _FakeTextOnlyAdapter:
    """Adapter stub with no send_exec_approval — exercises the text fallback."""

    typed_command_prefix = "!"

    def __init__(self):
        self.text_calls = []

    async def send(self, chat_id, content, metadata=None):
        self.text_calls.append((chat_id, content, metadata))
        return SendResult(success=True)


class _FakeRunner:
    def __init__(self, adapters=None, profile_adapters=None):
        self.adapters = adapters or {}
        self._profile_adapters = profile_adapters or {}


# ---------------------------------------------------------------------------
# Pure helpers: target parsing / config resolution
# ---------------------------------------------------------------------------

class TestParseDelegateTarget:
    def test_valid_target(self):
        assert _parse_delegate_target("slack:C0B8JK868SX") == ("slack", "C0B8JK868SX")

    def test_lowercases_platform_name(self):
        assert _parse_delegate_target("Slack:C0B8JK868SX") == ("slack", "C0B8JK868SX")

    def test_no_colon_is_malformed(self):
        assert _parse_delegate_target("slack") is None

    def test_empty_string_is_malformed(self):
        assert _parse_delegate_target("") is None

    def test_missing_platform_is_malformed(self):
        assert _parse_delegate_target(":C0B8JK868SX") is None

    def test_missing_chat_id_is_malformed(self):
        assert _parse_delegate_target("slack:") is None

    def test_none_is_malformed(self):
        assert _parse_delegate_target(None) is None


class TestResolveDelegateTarget:
    def test_per_route_wins_over_global(self):
        route_config = {"approval_delegate": "slack:route-channel"}
        global_approvals = {"delegate": "slack:global-channel"}
        assert _resolve_delegate_target(route_config, global_approvals) == "slack:route-channel"

    def test_falls_back_to_global_default(self):
        route_config = {}
        global_approvals = {"delegate": "slack:global-channel"}
        assert _resolve_delegate_target(route_config, global_approvals) == "slack:global-channel"

    def test_neither_configured_returns_none(self):
        assert _resolve_delegate_target({}, {}) is None

    def test_empty_string_route_value_falls_back_to_global(self):
        route_config = {"approval_delegate": ""}
        global_approvals = {"delegate": "slack:global-channel"}
        assert _resolve_delegate_target(route_config, global_approvals) == "slack:global-channel"


class TestResolveDelegateTimeout:
    def test_no_override_configured_returns_none(self):
        assert _resolve_delegate_timeout({}, {}) is None

    def test_per_route_override_wins(self):
        route_config = {"approval_delegate_timeout_seconds": 600}
        global_approvals = {"headless_timeout_seconds": 900}
        assert _resolve_delegate_timeout(route_config, global_approvals) == 600

    def test_falls_back_to_global_default(self):
        global_approvals = {"headless_timeout_seconds": 900}
        assert _resolve_delegate_timeout({}, global_approvals) == 900

    def test_non_numeric_value_is_ignored(self):
        route_config = {"approval_delegate_timeout_seconds": "not-a-number"}
        assert _resolve_delegate_timeout(route_config, {}) is None

    def test_zero_value_is_ignored(self):
        """A 0 override would make _await_gateway_decision's deadline math
        (max(timeout, 0)) deny instantly instead of waiting — must be
        rejected as malformed config, not honored (Copilot review #2)."""
        route_config = {"approval_delegate_timeout_seconds": 0}
        assert _resolve_delegate_timeout(route_config, {}) is None

    def test_negative_value_is_ignored(self):
        route_config = {"approval_delegate_timeout_seconds": -30}
        assert _resolve_delegate_timeout(route_config, {}) is None

    def test_zero_global_default_is_also_ignored(self):
        global_approvals = {"headless_timeout_seconds": 0}
        assert _resolve_delegate_timeout({}, global_approvals) is None


class TestGetDelegateAdapter:
    def test_finds_adapter_in_default_adapters(self):
        adapter = _FakeButtonAdapter()
        runner = _FakeRunner(adapters={Platform.SLACK: adapter})
        assert _get_delegate_adapter(runner, "slack") is adapter

    def test_falls_back_to_profile_adapters(self):
        adapter = _FakeButtonAdapter()
        runner = _FakeRunner(profile_adapters={"work": {Platform.SLACK: adapter}})
        assert _get_delegate_adapter(runner, "slack") is adapter

    def test_unconnected_platform_returns_none(self):
        runner = _FakeRunner(adapters={})
        assert _get_delegate_adapter(runner, "slack") is None

    def test_unknown_platform_name_returns_none(self):
        runner = _FakeRunner(adapters={Platform.SLACK: _FakeButtonAdapter()})
        assert _get_delegate_adapter(runner, "not-a-real-platform") is None


# ---------------------------------------------------------------------------
# maybe_register_delegate — fail-closed / no-op paths (Scenario: no delegate
# configured; Scenario: delegate configured but unreachable, from issue #1)
# ---------------------------------------------------------------------------

class TestMaybeRegisterDelegateNoOpPaths:
    SESSION_KEY = "webhook:github:test-delivery-noop"

    def teardown_method(self):
        from tools import approval as mod
        mod.unregister_gateway_notify(self.SESSION_KEY)

    @pytest.mark.asyncio
    async def test_no_target_configured_is_noop(self):
        runner = _FakeRunner(adapters={Platform.SLACK: _FakeButtonAdapter()})
        registered = maybe_register_delegate(self.SESSION_KEY, {}, runner, {})
        assert registered is False

        from tools import approval as mod
        assert self.SESSION_KEY not in mod._gateway_notify_cbs

    @pytest.mark.asyncio
    async def test_malformed_target_is_noop(self):
        runner = _FakeRunner(adapters={Platform.SLACK: _FakeButtonAdapter()})
        route_config = {"approval_delegate": "not-a-valid-target"}
        registered = maybe_register_delegate(self.SESSION_KEY, route_config, runner, {})
        assert registered is False

    @pytest.mark.asyncio
    async def test_no_gateway_runner_is_noop(self):
        route_config = {"approval_delegate": "slack:C0B8JK868SX"}
        registered = maybe_register_delegate(self.SESSION_KEY, route_config, None, {})
        assert registered is False

    @pytest.mark.asyncio
    async def test_unconnected_platform_is_noop(self):
        runner = _FakeRunner(adapters={})  # slack not connected
        route_config = {"approval_delegate": "slack:C0B8JK868SX"}
        registered = maybe_register_delegate(self.SESSION_KEY, route_config, runner, {})
        assert registered is False

    def test_no_running_event_loop_is_noop(self):
        """maybe_register_delegate requires a running loop (it captures one
        for the notify_cb to schedule sends on later from the agent thread).
        Called synchronously, outside any event loop, it must no-op rather
        than raise."""
        runner = _FakeRunner(adapters={Platform.SLACK: _FakeButtonAdapter()})
        route_config = {"approval_delegate": "slack:C0B8JK868SX"}
        registered = maybe_register_delegate(self.SESSION_KEY, route_config, runner, {})
        assert registered is False


# ---------------------------------------------------------------------------
# maybe_register_delegate — successful registration + notify_cb behavior
# (Scenario: delegate configured, human approves / denies, from issue #1)
# ---------------------------------------------------------------------------

class TestMaybeRegisterDelegateSuccess:
    SESSION_KEY = "webhook:github:test-delivery-success"

    def teardown_method(self):
        from tools import approval as mod
        mod.unregister_gateway_notify(self.SESSION_KEY)

    @pytest.mark.asyncio
    async def test_registers_notify_cb_when_target_configured(self):
        adapter = _FakeButtonAdapter()
        runner = _FakeRunner(adapters={Platform.SLACK: adapter})
        route_config = {"approval_delegate": "slack:C0B8JK868SX"}

        registered = maybe_register_delegate(self.SESSION_KEY, route_config, runner, {})
        assert registered is True

        from tools import approval as mod
        assert self.SESSION_KEY in mod._gateway_notify_cbs

    @pytest.mark.asyncio
    async def test_registration_is_pinned(self):
        """A delegate registration must be pinned, or gateway/run.py's later
        unconditional per-turn registration of its own default notify_cb
        silently clobbers it (Copilot review #2, finding 4)."""
        adapter = _FakeButtonAdapter()
        runner = _FakeRunner(adapters={Platform.SLACK: adapter})
        route_config = {"approval_delegate": "slack:C0B8JK868SX"}
        maybe_register_delegate(self.SESSION_KEY, route_config, runner, {})

        from tools import approval as mod
        assert self.SESSION_KEY in mod._gateway_notify_pinned

    @pytest.mark.asyncio
    async def test_pinned_registration_survives_a_later_unpinned_registration(self):
        """Simulates gateway/run.py's per-turn _run_agent_inner call, which
        unconditionally calls register_gateway_notify(session_key, its own
        default cb) with no pinning, moments after this delegate registers."""
        adapter = _FakeButtonAdapter()
        runner = _FakeRunner(adapters={Platform.SLACK: adapter})
        route_config = {"approval_delegate": "slack:C0B8JK868SX"}
        maybe_register_delegate(self.SESSION_KEY, route_config, runner, {})

        from tools import approval as mod
        delegate_cb = mod._gateway_notify_cbs[self.SESSION_KEY]

        def _run_py_default_cb(approval_data):
            raise AssertionError("the native default cb ran instead of the delegate")

        mod.register_gateway_notify(self.SESSION_KEY, _run_py_default_cb)  # unpinned

        assert mod._gateway_notify_cbs[self.SESSION_KEY] is delegate_cb
        assert self.SESSION_KEY in mod._gateway_notify_pinned

    @pytest.mark.asyncio
    async def test_global_default_target_is_used_when_no_route_override(self):
        adapter = _FakeButtonAdapter()
        runner = _FakeRunner(adapters={Platform.SLACK: adapter})
        global_approvals = {"delegate": "slack:C0B8JK868SX"}

        registered = maybe_register_delegate(self.SESSION_KEY, {}, runner, global_approvals)
        assert registered is True

    @pytest.mark.asyncio
    async def test_notify_cb_prefers_button_based_send_exec_approval(self):
        adapter = _FakeButtonAdapter()
        runner = _FakeRunner(adapters={Platform.SLACK: adapter})
        route_config = {"approval_delegate": "slack:C0B8JK868SX"}
        maybe_register_delegate(self.SESSION_KEY, route_config, runner, {})

        from tools import approval as mod
        cb = mod._gateway_notify_cbs[self.SESSION_KEY]
        # cb() blocks synchronously on a future scheduled back onto this
        # event loop (mirrors how the agent's worker thread calls it in
        # production) — must run it off-thread so the loop stays free to
        # actually service that scheduled coroutine.
        await asyncio.to_thread(cb, {"command": "rm -rf /tmp/foo", "description": "cleanup"})

        assert len(adapter.approval_calls) == 1
        call = adapter.approval_calls[0]
        assert call["chat_id"] == "C0B8JK868SX"
        assert call["session_key"] == self.SESSION_KEY
        assert call["command"] == "rm -rf /tmp/foo"
        assert not adapter.text_calls  # button path succeeded, no fallback

    @pytest.mark.asyncio
    async def test_notify_cb_falls_back_to_text_when_adapter_has_no_button_support(self):
        adapter = _FakeTextOnlyAdapter()
        runner = _FakeRunner(adapters={Platform.SLACK: adapter})
        route_config = {"approval_delegate": "slack:C0B8JK868SX"}
        maybe_register_delegate(self.SESSION_KEY, route_config, runner, {})

        from tools import approval as mod
        cb = mod._gateway_notify_cbs[self.SESSION_KEY]
        await asyncio.to_thread(cb, {"command": "rm -rf /tmp/foo", "description": "cleanup"})

        assert len(adapter.text_calls) == 1
        chat_id, message, _metadata = adapter.text_calls[0]
        assert chat_id == "C0B8JK868SX"
        assert "rm -rf /tmp/foo" in message
        # Uses the adapter's own typed_command_prefix ("!"), not a hardcoded "/".
        assert "!approve" in message

    @pytest.mark.asyncio
    async def test_notify_cb_falls_back_to_text_when_button_send_fails(self):
        adapter = _FakeButtonAdapter(approval_success=False)
        runner = _FakeRunner(adapters={Platform.SLACK: adapter})
        route_config = {"approval_delegate": "slack:C0B8JK868SX"}
        maybe_register_delegate(self.SESSION_KEY, route_config, runner, {})

        from tools import approval as mod
        cb = mod._gateway_notify_cbs[self.SESSION_KEY]
        await asyncio.to_thread(cb, {"command": "rm -rf /tmp/foo", "description": "cleanup"})

        assert len(adapter.approval_calls) == 1  # tried the button path first
        assert len(adapter.text_calls) == 1       # then fell back to text


# ---------------------------------------------------------------------------
# tools.approval's pinned-registration guard (Copilot review #2, finding 4):
# gateway/run.py's per-turn agent runner unconditionally re-registers its
# own default (unpinned) notify_cb for every session on every turn. Without
# pinning, that clobbers a delegate registered moments earlier — silently,
# no exception, the feature just does nothing. These tests exercise the
# registry directly, independent of gateway/approval_delegate.py.
# ---------------------------------------------------------------------------

class TestPinnedRegistrationGuard:
    SESSION_KEY = "test-pinned-registration-session"

    def teardown_method(self):
        from tools import approval as mod
        mod.unregister_gateway_notify(self.SESSION_KEY)

    def test_unpinned_registration_is_pinned_false_by_default(self):
        from tools import approval as mod
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None)
        assert self.SESSION_KEY not in mod._gateway_notify_pinned

    def test_pinned_registration_marks_session_pinned(self):
        from tools import approval as mod
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None, pinned=True)
        assert self.SESSION_KEY in mod._gateway_notify_pinned

    def test_unpinned_call_does_not_overwrite_a_pinned_cb(self):
        from tools import approval as mod
        pinned_cb = lambda data: None
        mod.register_gateway_notify(self.SESSION_KEY, pinned_cb, pinned=True)
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None)  # unpinned
        assert mod._gateway_notify_cbs[self.SESSION_KEY] is pinned_cb
        assert self.SESSION_KEY in mod._gateway_notify_pinned

    def test_unpinned_call_does_not_clear_pinned_timeout_override(self):
        from tools import approval as mod
        mod.register_gateway_notify(
            self.SESSION_KEY, lambda data: None, timeout_override=900, pinned=True,
        )
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None)  # unpinned, no-op
        assert mod._get_approval_timeout(self.SESSION_KEY) == 900

    def test_pinned_call_can_overwrite_a_pinned_cb(self):
        from tools import approval as mod
        first_cb = lambda data: None
        second_cb = lambda data: None
        mod.register_gateway_notify(self.SESSION_KEY, first_cb, pinned=True)
        mod.register_gateway_notify(self.SESSION_KEY, second_cb, pinned=True)
        assert mod._gateway_notify_cbs[self.SESSION_KEY] is second_cb

    def test_unpinned_call_overwrites_a_prior_unpinned_cb_as_before(self):
        """Normal (non-delegate) behavior is unchanged: each turn's fresh
        unpinned registration still wins over the previous turn's."""
        from tools import approval as mod
        first_cb = lambda data: None
        second_cb = lambda data: None
        mod.register_gateway_notify(self.SESSION_KEY, first_cb)
        mod.register_gateway_notify(self.SESSION_KEY, second_cb)
        assert mod._gateway_notify_cbs[self.SESSION_KEY] is second_cb

    def test_unregister_clears_pinned_flag(self):
        from tools import approval as mod
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None, pinned=True)
        mod.unregister_gateway_notify(self.SESSION_KEY)
        assert self.SESSION_KEY not in mod._gateway_notify_pinned
        # A fresh unpinned registration after unregister must succeed
        # normally — the pin from a prior (now-ended) session must not leak.
        new_cb = lambda data: None
        mod.register_gateway_notify(self.SESSION_KEY, new_cb)
        assert mod._gateway_notify_cbs[self.SESSION_KEY] is new_cb


# ---------------------------------------------------------------------------
# Session-scoped timeout override (design doc Part 3a extension to
# tools/approval.py's _get_approval_timeout)
# ---------------------------------------------------------------------------

class TestSessionScopedTimeoutOverride:
    SESSION_KEY = "webhook:github:test-delivery-timeout"

    def teardown_method(self):
        from tools import approval as mod
        mod.unregister_gateway_notify(self.SESSION_KEY)

    def test_register_with_timeout_override_is_used_by_get_approval_timeout(self):
        from tools import approval as mod
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None, timeout_override=900)
        assert mod._get_approval_timeout(self.SESSION_KEY) == 900

    def test_no_override_falls_back_to_global_default(self):
        from tools import approval as mod
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None)
        # No override registered — falls back to whatever the global config
        # resolves to (default 300), not some stale prior-test value.
        assert self.SESSION_KEY not in mod._gateway_notify_timeouts

    def test_unrelated_session_key_is_unaffected(self):
        from tools import approval as mod
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None, timeout_override=900)
        assert mod._get_approval_timeout("some-other-session") != 900

    def test_unregister_clears_the_override(self):
        from tools import approval as mod
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None, timeout_override=900)
        mod.unregister_gateway_notify(self.SESSION_KEY)
        assert self.SESSION_KEY not in mod._gateway_notify_timeouts

    @pytest.mark.asyncio
    async def test_maybe_register_delegate_wires_configured_timeout_override(self):
        adapter = _FakeButtonAdapter()
        runner = _FakeRunner(adapters={Platform.SLACK: adapter})
        route_config = {
            "approval_delegate": "slack:C0B8JK868SX",
            "approval_delegate_timeout_seconds": 900,
        }
        maybe_register_delegate(self.SESSION_KEY, route_config, runner, {})

        from tools import approval as mod
        assert mod._get_approval_timeout(self.SESSION_KEY) == 900
