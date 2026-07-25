"""Opt-in approval delegation for headless gateway sessions.

Headless triggers (webhook today, cron as a documented future follow-up)
have no interactive UI for a human to respond to a ``approvals.mode: smart``
ambiguous-command prompt through — ``tools.approval.register_gateway_notify``
is simply never called for them, so every ambiguous command hangs for the
full approval timeout and then fails closed, one at a time, with nobody able
to say yes.

This module lets an operator opt a headless session into forwarding those
prompts to a real interactive platform (e.g. Slack) and having a human
resolve them there, through that platform's own existing **button-based**
``send_exec_approval`` mechanism — the same one already used for native chat
sessions on that platform. No platform-specific resolver code is added:
adapters resolve a button click generically by whatever ``session_key`` was
embedded in the button payload at send time, regardless of where that
session originated (confirmed for Slack's adapter, which treats
``session_key`` as an opaque pass-through both when rendering the approval
buttons and when resolving a click against
``tools.approval.resolve_gateway_approval``).

Delegation is **button-only by design**. A plain-text ``/approve`` reply in
the delegate chat is resolved against that chat's *own* session key — it
cannot reach the headless session's approval, and could even resolve an
unrelated approval already pending in that chat. So an adapter without
``send_exec_approval`` gets a loud log and a no-op (fail-closed), not a
broken text prompt. Text-based cross-session resolution would need its own
explicit token/command design and is deliberately out of scope here.

Three details matter for correctness and are easy to get wrong:

1. **The session_key must be the real gateway approval key**, i.e. whatever
   ``gateway.session.build_session_key(source, ...)`` computes for the
   triggering event's ``source`` — not a platform's own delivery/chat
   identifier (e.g. a raw webhook ``chat_id``). ``tools.approval`` waits on
   the ``build_session_key``-derived key; registering under anything else
   means the notify_cb is never found.
2. **The registration must be pinned** (``register_gateway_notify(...,
   pinned=True)``). ``gateway/run.py``'s per-turn agent runner
   unconditionally re-registers its own default (unpinned) notify_cb for
   *every* session on *every* turn, including headless ones, moments after
   a caller like ``gateway/platforms/webhook.py`` registers a delegate.
   Without pinning, that later unpinned call silently overwrites the
   delegate — no exception, the feature just quietly does nothing.
3. **"Always Allow" is never offered on a delegated prompt**
   (``allow_permanent=False``, unconditionally). A permanent approval writes
   a disk-persisted, process-global allowlist entry, and a human approving
   out-of-band has no visibility into what future webhook payloads that
   pattern would then auto-approve. Session scope is the ceiling.

Fail-closed behavior is unchanged in every case this module doesn't
actively improve: no delegate configured, a malformed target, an
unconnected/unknown platform, a button-less adapter, or a delivery failure
all leave the session behaving exactly as it does today (denied after the
approval timeout).

Be precise about what this changes: the Smart DENY classifier's *logic* is
untouched — outright-dangerous commands are still auto-denied with no human
in the loop — but for the *ambiguous* bucket (which can include
dangerous-looking commands the classifier couldn't condemn outright), the
outcome genuinely changes from "guaranteed timeout-deny" to
"human-approvable via one click in the delegate chat." That is the point of
the feature, and operators opting in should understand they are granting
that bucket a real approval path, gated by the delegate platform's own
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


def _parse_delegate_target(target: Optional[str]) -> Optional[Tuple[str, str]]:
    """Split ``"slack:C0B8JK868SX"`` into ``("slack", "C0B8JK868SX")``.

    Returns None if *target* isn't in ``<platform>:<chat-or-channel-id>``
    form. Non-string values (a YAML ``true``, a bare number, …) are malformed
    config, not an exception path — they must fail closed like any other bad
    target, never turn a webhook request into a TypeError.
    """
    if not isinstance(target, str) or not target or ":" not in target:
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
    """Per-route override wins; else the global ``approvals.headless_timeout_seconds`` default.

    Non-positive values (``0`` or negative) are treated as malformed config
    and ignored, falling back to the global ``approvals.timeout`` default,
    rather than being honored — a ``0``/negative override would otherwise
    turn ``_await_gateway_decision``'s deadline math (``max(timeout, 0)``)
    into an immediate deny with no real wait, defeating the point of
    configuring a delegate at all.
    """
    value = route_config.get("approval_delegate_timeout_seconds")
    if value is None:
        value = global_approvals.get("headless_timeout_seconds")
    if value is None:
        return None
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric approval delegate timeout: %r", value)
        return None
    if timeout <= 0:
        logger.warning(
            "Ignoring non-positive approval delegate timeout: %r "
            "(falling back to the global approvals.timeout default)",
            value,
        )
        return None
    return timeout


def _get_delegate_adapter(
    gateway_runner: Any, platform_name: str, profile: Optional[str] = None,
) -> Any:
    """Look up a live adapter instance for *platform_name*.

    Profile-scoped, unlike ``WebhookAdapter._deliver_cross_platform``'s
    scan-everything lookup: a session stamped with a secondary *profile*
    resolves ONLY within that profile's adapter map, and an unstamped
    session resolves ONLY among the default adapters. Falling back to some
    other profile's bot would send the approval prompt through the wrong
    workspace/account — for an approval surface that's a fail-closed
    situation, not a convenience fallback.
    """
    from gateway.config import Platform

    try:
        target_platform = Platform(platform_name)
    except ValueError:
        return None

    if profile:
        profile_adapters = getattr(gateway_runner, "_profile_adapters", None) or {}
        amap = profile_adapters.get(profile)
        if isinstance(amap, dict):
            return amap.get(target_platform)
        return None
    return gateway_runner.adapters.get(target_platform)


def maybe_register_delegate(
    session_key: str,
    route_config: dict,
    gateway_runner: Any,
    global_approvals: Optional[dict] = None,
    profile: Optional[str] = None,
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

    adapter = _get_delegate_adapter(gateway_runner, platform_name, profile=profile)
    if adapter is None:
        logger.warning(
            "approval_delegate=%r configured for session %s but platform %r "
            "is not connected — ignoring, falling back to default fail-closed behavior",
            target, session_key, platform_name,
        )
        return False

    # Delegation requires the target adapter's button-based send_exec_approval.
    # A plain-text prompt cannot work here: a typed /approve reply in the
    # delegate chat resolves against THAT chat's own session key, not the
    # headless session registered under *session_key* — only the button
    # payload carries the original session_key back to
    # resolve_gateway_approval. Worse, a well-meaning typed reply could
    # resolve an unrelated approval already pending in the delegate chat.
    # So: no buttons → honest no-op, fail closed, loud log. (Class-level
    # check, not instance — avoids MagicMock false positives in tests.)
    if getattr(type(adapter), "send_exec_approval", None) is None:
        logger.warning(
            "approval_delegate=%r configured for session %s but platform %r's "
            "adapter has no button-based send_exec_approval — text replies "
            "cannot resolve a delegated session's approval, so delegation is "
            "disabled for this session (fail-closed behavior unchanged)",
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
        Mirrors gateway/run.py's native ``_approval_notify_sync``, minus its
        text fallback: only the button payload carries *session_key* back to
        ``resolve_gateway_approval``, so if the button send fails there is
        nothing useful to say in text — log loudly and let the approval time
        out to its normal fail-closed denial.
        """
        from agent.redact import redact_sensitive_text

        cmd = redact_sensitive_text(str(approval_data.get("command", "") or ""), force=True)
        desc = approval_data.get("description", "dangerous command")

        try:
            fut = safe_schedule_threadsafe(
                adapter.send_exec_approval(
                    chat_id=chat_id,
                    command=cmd,
                    session_key=session_key,
                    description=desc,
                    # Never offer "Always Allow" on a delegated prompt: it
                    # writes a disk-persisted, process-global allowlist entry,
                    # and the human approving out-of-band has no visibility
                    # into what future webhook payloads that pattern would
                    # then auto-approve. Session scope is the ceiling here.
                    allow_permanent=False,
                    allow_session=approval_data.get("allow_session", True),
                    smart_denied=approval_data.get("smart_denied", False),
                ),
                loop,
                logger=logger,
                log_message="Delegated send_exec_approval scheduling error",
            )
            if fut is None:
                raise RuntimeError("send_exec_approval: loop unavailable")
            try:
                result = fut.result(timeout=15)
            except Exception:
                # Cancel the still-scheduled coroutine so a slow send can't
                # land a late, orphaned prompt after we've given up on it.
                # cancel() is a no-op if the future already finished/failed.
                fut.cancel()
                raise
            if not result.success:
                logger.error(
                    "Delegated approval prompt could not be delivered to %s "
                    "(send returned error: %s) — approval will time out and "
                    "fail closed for session %s",
                    target, result.error, session_key,
                )
        except Exception as exc:
            logger.error(
                "Delegated approval prompt could not be delivered to %s (%s) — "
                "approval will time out and fail closed for session %s",
                target, exc, session_key,
            )

    from tools.approval import register_gateway_notify

    timeout_override = _resolve_delegate_timeout(route_config, global_approvals)
    # pinned=True: gateway/run.py's per-turn agent runner unconditionally
    # re-registers its own default (unpinned) notify_cb for every session on
    # every turn, including this one, moments after this call returns. A
    # pinned registration refuses that overwrite (see
    # tools.approval.register_gateway_notify's docstring) — without it, the
    # delegate would be silently clobbered before the agent's first
    # approval-worthy command ever runs.
    register_gateway_notify(
        session_key, _delegate_notify_sync, timeout_override=timeout_override, pinned=True,
    )
    logger.info(
        "Registered approval delegate %s for session %s%s",
        target, session_key,
        f" (timeout override {timeout_override}s)" if timeout_override else "",
    )
    return True
