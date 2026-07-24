"""Opt-in approval delegation for headless gateway sessions.

Headless triggers (webhook today, cron as a documented future follow-up)
have no interactive UI for a human to respond to a ``approvals.mode: smart``
ambiguous-command prompt through — ``tools.approval.register_gateway_notify``
is simply never called for them, so every ambiguous command hangs for the
full approval timeout and then fails closed, one at a time, with nobody able
to say yes.

This module lets an operator opt a headless session into forwarding those
prompts to a real interactive platform (e.g. Slack) and having a human
resolve them there, through that platform's own existing approve/deny
mechanism (button-based ``send_exec_approval`` where the adapter supports
it, else the existing plain-text fallback) — the same mechanism already used
for native chat sessions on that platform. No platform-specific resolver
code is added: adapters resolve an inbound approval reply generically by
whatever ``session_key`` was embedded at send time, regardless of where that
session originated (confirmed for Slack's adapter, which treats
``session_key`` as an opaque pass-through both when rendering the approval
buttons and when resolving a click against
``tools.approval.resolve_gateway_approval``).

Fail-closed behavior is unchanged in every case this module doesn't
actively improve: no delegate configured, a malformed target, an
unconnected/unknown platform, or a delivery failure all leave the session
behaving exactly as it does today (denied after the approval timeout). This
module only ever adds a chance for a human to say yes — it never loosens
the Smart DENY classifier, and it never bypasses the target platform's own
inbound authorization (SECURITY.md §2.6: a session_key is a routing handle,
not an authorization boundary — the delegate relies entirely on the target
adapter's existing authorization check, which runs before approval
resolution regardless of where the session_key originated).

See ``hermes-webhook-approval-delegate-design.md`` (Part 3a) for the full
design and its reasoning trail.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Tuple

from agent.async_utils import safe_schedule_threadsafe

logger = logging.getLogger(__name__)


def _parse_delegate_target(target: str) -> Optional[Tuple[str, str]]:
    """Split ``"slack:C0B8JK868SX"`` into ``("slack", "C0B8JK868SX")``.

    Returns None if *target* isn't in ``<platform>:<chat-or-channel-id>``
    form.
    """
    if not target or ":" not in target:
        return None
    platform_name, _, chat_id = target.partition(":")
    platform_name = platform_name.strip().lower()
    chat_id = chat_id.strip()
    if not platform_name or not chat_id:
        return None
    return platform_name, chat_id


def _resolve_delegate_target(route_config: dict, global_approvals: dict) -> Optional[str]:
    """Per-route ``approval_delegate`` wins; else the global ``approvals.delegate`` default."""
    target = route_config.get("approval_delegate")
    if not target:
        target = global_approvals.get("delegate")
    return target or None


def _resolve_delegate_timeout(route_config: dict, global_approvals: dict) -> Optional[int]:
    """Per-route override wins; else the global ``approvals.headless_timeout_seconds`` default."""
    value = route_config.get("approval_delegate_timeout_seconds")
    if value is None:
        value = global_approvals.get("headless_timeout_seconds")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric approval delegate timeout: %r", value)
        return None


def _get_delegate_adapter(gateway_runner: Any, platform_name: str) -> Any:
    """Look up a live adapter instance for *platform_name*.

    Mirrors ``WebhookAdapter._deliver_cross_platform``'s adapter lookup
    (default adapters first, falling back to per-profile adapters) so a
    delegate target resolves the same way a webhook ``deliver:`` target
    already does — deliberately reusing that lookup shape rather than
    inventing a second one.
    """
    from gateway.config import Platform

    try:
        target_platform = Platform(platform_name)
    except ValueError:
        return None

    adapter = gateway_runner.adapters.get(target_platform)
    if adapter is not None:
        return adapter
    for amap in (getattr(gateway_runner, "_profile_adapters", None) or {}).values():
        if not isinstance(amap, dict):
            continue
        candidate = amap.get(target_platform)
        if candidate is not None:
            return candidate
    return None


def maybe_register_delegate(
    session_key: str,
    route_config: dict,
    gateway_runner: Any,
    global_approvals: Optional[dict] = None,
) -> bool:
    """Register an approval delegate for *session_key* if one is configured.

    No-op (returns False, behavior unchanged from today) unless a delegate
    target is configured — per-route via ``route_config["approval_delegate"]``,
    or globally via ``approvals.delegate``. Also no-ops (with a warning) on a
    malformed target, a missing/disconnected gateway_runner, an unrecognized
    platform name, or an unconnected target platform — always falling back
    to the existing fail-closed behavior rather than raising.

    Returns True if a delegate was actually registered. Callers should call
    ``tools.approval.unregister_gateway_notify(session_key)`` when the
    session ends regardless of this return value (that call is a no-op/safe
    if nothing was registered).
    """
    if global_approvals is None:
        global_approvals = {}

    target = _resolve_delegate_target(route_config, global_approvals)
    if not target:
        return False

    parsed = _parse_delegate_target(target)
    if parsed is None:
        logger.warning(
            'Malformed approval_delegate target %r for session %s '
            '(expected "<platform>:<chat-or-channel-id>") — ignoring, '
            "falling back to default fail-closed behavior",
            target, session_key,
        )
        return False
    platform_name, chat_id = parsed

    if gateway_runner is None:
        logger.warning(
            "approval_delegate=%r configured for session %s but no gateway_runner "
            "is available — ignoring, falling back to default fail-closed behavior",
            target, session_key,
        )
        return False

    adapter = _get_delegate_adapter(gateway_runner, platform_name)
    if adapter is None:
        logger.warning(
            "approval_delegate=%r configured for session %s but platform %r "
            "is not connected — ignoring, falling back to default fail-closed behavior",
            target, session_key, platform_name,
        )
        return False

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning(
            "maybe_register_delegate called with no running event loop for "
            "session %s — ignoring, falling back to default fail-closed behavior",
            session_key,
        )
        return False

    def _delegate_notify_sync(approval_data: dict) -> None:
        """Bridge sync agent thread → event loop, targeting the delegate
        adapter/chat instead of the session's own originating adapter/chat.
        Mirrors gateway/run.py's native ``_approval_notify_sync``.
        """
        from agent.redact import redact_sensitive_text
        from gateway.run import _format_exec_approval_fallback

        cmd = redact_sensitive_text(str(approval_data.get("command", "") or ""), force=True)
        desc = approval_data.get("description", "dangerous command")

        # Prefer button-based approval when the delegate adapter supports it.
        # Check the *class* for the method, not the instance — avoids false
        # positives from MagicMock auto-attribute creation in tests.
        if getattr(type(adapter), "send_exec_approval", None) is not None:
            try:
                fut = safe_schedule_threadsafe(
                    adapter.send_exec_approval(
                        chat_id=chat_id,
                        command=cmd,
                        session_key=session_key,
                        description=desc,
                        allow_permanent=approval_data.get("allow_permanent", True),
                        allow_session=approval_data.get("allow_session", True),
                        smart_denied=approval_data.get("smart_denied", False),
                    ),
                    loop,
                    logger=logger,
                    log_message="Delegated send_exec_approval scheduling error",
                )
                if fut is None:
                    raise RuntimeError("send_exec_approval: loop unavailable")
                result = fut.result(timeout=15)
                if result.success:
                    return
                logger.warning(
                    "Delegated button-based approval failed (send returned error), "
                    "falling back to text: %s",
                    result.error,
                )
            except Exception as exc:
                logger.warning(
                    "Delegated button-based approval failed, falling back to text: %s", exc
                )

        prefix = getattr(adapter, "typed_command_prefix", "/")
        msg = _format_exec_approval_fallback(
            cmd, desc, prefix,
            allow_permanent=approval_data.get("allow_permanent", True),
            allow_session=approval_data.get("allow_session", True),
            smart_denied=approval_data.get("smart_denied", False),
        )
        try:
            send_fut = safe_schedule_threadsafe(
                adapter.send(chat_id, msg),
                loop,
                logger=logger,
                log_message="Delegated approval text-send scheduling error",
            )
            if send_fut is not None:
                send_fut.result(timeout=15)
        except Exception as exc:
            logger.error("Failed to send delegated approval request: %s", exc)

    from tools.approval import register_gateway_notify

    timeout_override = _resolve_delegate_timeout(route_config, global_approvals)
    register_gateway_notify(session_key, _delegate_notify_sync, timeout_override=timeout_override)
    logger.info(
        "Registered approval delegate %s for session %s%s",
        target, session_key,
        f" (timeout override {timeout_override}s)" if timeout_override else "",
    )
    return True
