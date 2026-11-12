import weakref
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from functools import lru_cache, reduce
from itertools import count
from types import MappingProxyType
from typing import Final, TypeAlias, TypeVar

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger, overrides_hook
from litellm.integrations.custom_logger_v2.call import CallFacts
from litellm.integrations.custom_logger_v2.dispatcher import (
    Between,
    HookResult,
    RequestScope,
    run_legacy_plan,
    run_legacy_plan_sync,
)
from litellm.integrations.custom_logger_v2.enums import Activation, Field, Hook
from litellm.integrations.custom_logger_v2.hooks import HOOK_SPECS, Subscription
from litellm.integrations.custom_logger_v2.host import NativeOutcome, current_scope, get_dispatcher
from litellm.integrations.custom_logger_v2.logger import CustomLoggerV2
from litellm.integrations.custom_logger_v2.manifest import CallbackManifest, LegacyBinding, build_manifest
from litellm.integrations.custom_logger_v2.outcomes import Abort, InvalidPatchError, MergeField, Patch, PatchOp, Respond
from litellm.integrations.custom_logger_v2.patches import LiveFields
from litellm.integrations.custom_logger_v2.planner import Entry, HookPlan, plan_hook
from litellm.integrations.custom_logger_v2.selection import RequestSelection, Selection

ResultT: Final = TypeVar("ResultT")

SyncLegacyBinding: TypeAlias = Callable[[object, object], object]

PARALLEL_GUARDRAIL_HOOKS: Final = frozenset({Hook.pre_call, Hook.post_call_success})
LEGACY_READS: Final = frozenset({Field.deprecated_kwargs, Field.deprecated_response_obj})
PROXY_REQUEST_KEYS: Final[Mapping[Field, str]] = MappingProxyType(
    {
        Field.model: "model",
        Field.messages: "messages",
        Field.input: "input",
        Field.tools: "tools",
    }
)
LINE_CACHE_SIZE: Final = 128
PLAN_CACHE_SIZE: Final = 64

_SERIALS: Final = count()
_ABSENT: Final = object()
_NO_SELECTION: Final = RequestSelection()


@lru_cache(maxsize=None)
def legacy_subscription(hook: Hook, parallel: bool) -> Subscription:
    spec: Final = HOOK_SPECS[hook]
    may_abort: Final = bool(spec.outcomes)
    gathered: Final = may_abort and (parallel or hook is Hook.during_call)
    return Subscription(
        hook=hook,
        handler="call",
        reads=LEGACY_READS,
        writes=frozenset({Field.deprecated_kwargs}) if spec.writes and not gathered else frozenset(),
        may_abort=may_abort,
        parallel=gathered,
        legacy=True,
    )


def _subscribes(cls: type, guardrail: bool, hook: Hook) -> bool:
    spec: Final = HOOK_SPECS[hook]
    if not spec.subscribable:
        return False
    if guardrail and spec.legacy_guardrail_events:
        return True
    return any(overrides_hook(cls, method) for method in spec.legacy_methods)


@lru_cache(maxsize=512)
def _derived_subscriptions(cls: type, guardrail: bool, parallel: bool) -> Mapping[Hook, Subscription]:
    return MappingProxyType(
        {
            hook: legacy_subscription(hook, parallel and hook in PARALLEL_GUARDRAIL_HOOKS)
            for hook in Hook
            if _subscribes(cls, guardrail, hook)
        }
    )


class LegacyLogger(CustomLoggerV2):
    def __init__(self, inner: CustomLogger) -> None:
        guardrail: Final = isinstance(inner, CustomGuardrail)
        self.parallel: Final = guardrail and bool(getattr(inner, "run_in_parallel", False))
        self.serial: Final = next(_SERIALS)
        self._inner: Final = weakref.ref(inner)
        self.manifest: Final = CallbackManifest(
            logger=self,
            selection=Selection(name=type(inner).__name__, activation=Activation.required),
            subscriptions=_derived_subscriptions(type(inner), guardrail, self.parallel),
            request_state=None,
            legacy_call=self.call,
        )

    @property
    def inner(self) -> CustomLogger | None:
        return self._inner()

    def subscription(self, hook: Hook, every_logger: bool) -> Subscription | None:
        derived: Final = self.manifest.subscriptions.get(hook)
        if derived is not None or not every_logger or not HOOK_SPECS[hook].subscribable:
            return derived
        return legacy_subscription(hook, self.parallel and hook in PARALLEL_GUARDRAIL_HOOKS)

    def call(self, binding: Callable[[object, object], ResultT], fields: Mapping[Field, object]) -> ResultT | None:
        inner: Final = self._inner()
        return None if inner is None else binding(inner, fields[Field.deprecated_kwargs])


_ADAPTERS: Final[dict[int, LegacyLogger]] = {}  # mutable-ok: adapters live as long as the logger they wrap


def adapter_for(inner: CustomLogger) -> LegacyLogger:
    key: Final = id(inner)
    cached: Final = _ADAPTERS.get(key)
    if cached is not None and cached.inner is inner:
        return cached
    adapter: Final = LegacyLogger(inner)
    _ADAPTERS[key] = adapter
    _ = weakref.finalize(inner, _ADAPTERS.pop, key, None)
    return adapter


class KwargsFields(LiveFields):
    __slots__ = ("_keys", "_kwargs", "_response")

    def __init__(self, kwargs: object, keys: Mapping[Field, str], response: object = _ABSENT) -> None:
        self._kwargs: Final = kwargs
        self._keys: Final = keys
        self._response: Final = response

    def _key(self, field: Field) -> str | None:
        key: Final = self._keys.get(field)
        return key if isinstance(self._kwargs, Mapping) and key in self._kwargs else None

    def __getitem__(self, field: Field) -> object:
        if field is Field.deprecated_kwargs:
            return self._kwargs
        if field is Field.deprecated_response_obj and self._response is not _ABSENT:
            return self._response
        key: Final = self._key(field)
        if key is None or not isinstance(self._kwargs, Mapping):
            raise KeyError(field)
        return _mapping_value(self._kwargs, key)

    def __iter__(self) -> Iterator[Field]:
        return iter(tuple(field for field in Field if self._has(field)))

    def __len__(self) -> int:
        return sum(1 for field in Field if self._has(field))

    def _has(self, field: Field) -> bool:
        if field is Field.deprecated_kwargs:
            return True
        if field is Field.deprecated_response_obj:
            return self._response is not _ABSENT
        return self._key(field) is not None

    def apply(self, patch: Patch) -> Mapping[Field, object]:
        return reduce(_apply_op, patch.ops, self)

    def replaced(self, field: Field, value: object) -> "KwargsFields":
        if field is Field.deprecated_kwargs:
            return KwargsFields(value, self._keys, self._response)
        return KwargsFields(self._kwargs, self._keys, value)

    def write(self, field: Field, value: object) -> None:
        target: Final = self._target(field)
        target[self._keys[field]] = value

    def merge(self, field: Field, keys: Mapping[str, object]) -> None:
        target: Final = self._target(field)
        current: Final = target.get(self._keys[field])
        if current is None:
            target[self._keys[field]] = dict(keys)
            return
        if not isinstance(current, MutableMapping):
            raise InvalidPatchError(f"{field.value} holds no mapping here, so keys cannot be merged into it")
        _update(current, keys)

    def _target(self, field: Field) -> MutableMapping[str, object]:
        if field not in self._keys:
            raise InvalidPatchError(f"{field.value} has no place in the legacy kwargs of this hook")
        if not isinstance(self._kwargs, MutableMapping):
            raise InvalidPatchError(f"{field.value} cannot be written because the legacy kwargs are not a mapping")
        return _string_keyed(self._kwargs)


def _mapping_value(mapping: Mapping[object, object], key: str) -> object:
    return mapping[key]


def _string_keyed(mapping: MutableMapping[str, object]) -> MutableMapping[str, object]:
    return mapping


def _update(target: MutableMapping[str, object], keys: Mapping[str, object]) -> None:
    target.update(keys)


def _apply_op(fields: Mapping[Field, object], op: PatchOp) -> Mapping[Field, object]:
    if not isinstance(fields, KwargsFields):
        raise InvalidPatchError("legacy kwargs patches need KwargsFields")
    if op.field in (Field.deprecated_kwargs, Field.deprecated_response_obj):
        if isinstance(op, MergeField):
            raise InvalidPatchError(f"{op.field.value} is replaced whole, never merged")
        return fields.replaced(op.field, op.value)
    if isinstance(op, MergeField):
        fields.merge(op.field, op.keys)
        return fields
    fields.write(op.field, op.value)
    return fields


def kwargs_patch(before: object, after: object) -> Patch | None:
    return None if after is before else Patch().set(Field.deprecated_kwargs, after)


@dataclass(frozen=True, slots=True, eq=False)
class HookLine:
    hook: Hook
    entries: tuple[Entry, ...]
    plan: HookPlan
    has_natives: bool
    plans: Callable[[tuple[int, ...]], HookPlan]

    @property
    def legacy(self) -> tuple[CustomLogger, ...]:
        return tuple(
            inner
            for entry in self.entries
            if isinstance(entry.manifest.logger, LegacyLogger) and (inner := entry.manifest.logger.inner) is not None
        )

    def plan_for(self, scope: RequestScope) -> HookPlan:
        selected: Final = tuple(
            index for index, entry in enumerate(self.entries) if entry.subscription.legacy or scope.selects(entry)
        )
        return self.plan if len(selected) == len(self.entries) else self.plans(selected)


def _native_entry(native: CustomLoggerV2, hook: Hook) -> tuple[CallbackManifest, Subscription] | None:
    manifest: Final = build_manifest(native)
    subscription: Final = manifest.subscriptions.get(hook)
    return None if subscription is None else (manifest, subscription)


def _legacy_entry(inner: CustomLogger, hook: Hook, every_logger: bool) -> tuple[CallbackManifest, Subscription] | None:
    adapter: Final = adapter_for(inner)
    subscription: Final = adapter.subscription(hook, every_logger)
    return None if subscription is None else (adapter.manifest, subscription)


def _line_member(
    item: object, hook: Hook, every_logger: bool, natives: bool
) -> tuple[CallbackManifest, Subscription] | None:
    if isinstance(item, CustomLoggerV2):
        return _native_entry(item, hook) if natives else None
    if isinstance(item, CustomLogger):
        return _legacy_entry(item, hook, every_logger)
    return None


def _build_line(hook: Hook, callbacks: Sequence[object], every_logger: bool, natives: bool) -> HookLine:
    members: Final = tuple(
        member for item in callbacks if (member := _line_member(item, hook, every_logger, natives)) is not None
    )
    entries: Final = tuple(
        Entry(position, manifest, subscription) for position, (manifest, subscription) in enumerate(members)
    )

    def subset_plan(selected: tuple[int, ...]) -> HookPlan:
        return plan_hook(tuple(entries[index] for index in selected))

    return HookLine(
        hook=hook,
        entries=entries,
        plan=plan_hook(entries),
        has_natives=any(not entry.subscription.legacy for entry in entries),
        plans=lru_cache(maxsize=PLAN_CACHE_SIZE)(subset_plan),
    )


@dataclass(frozen=True, slots=True, eq=False)
class _CachedLine:
    line: HookLine
    pinned: tuple[object, ...]


_LINES: Final[dict[tuple[object, ...], _CachedLine]] = {}  # mutable-ok: bounded cache of resolved hook lines


def _pinned(item: object) -> bool:
    return not isinstance(item, CustomLogger) or isinstance(item, CustomLoggerV2)


def _cache_line(key: tuple[object, ...], line: HookLine, callbacks: Sequence[object]) -> None:
    if len(_LINES) >= LINE_CACHE_SIZE:
        _LINES.clear()
    _LINES[key] = _CachedLine(line=line, pinned=tuple(item for item in callbacks if _pinned(item)))
    for item in callbacks:
        if not _pinned(item):
            _ = weakref.finalize(item, _LINES.pop, key, None)


def hook_line(hook: Hook, callbacks: Sequence[object], *, every_logger: bool = False, natives: bool = True) -> HookLine:
    key: Final = (hook, every_logger, natives, *map(id, callbacks))
    cached: Final = _LINES.get(key)
    if cached is not None:
        return cached.line
    line: Final = _build_line(hook, callbacks, every_logger, natives)
    _cache_line(key, line, callbacks)
    return line


def clear_line_cache_for_tests() -> None:
    _LINES.clear()


async def run_line(
    line: HookLine,
    fields: Mapping[Field, object],
    binding: LegacyBinding,
    *,
    between: Between | None = None,
    call: CallFacts | None = None,
    scope: RequestScope | None = None,
) -> HookResult[None]:
    if not line.has_natives:
        return HookResult(fields=await run_legacy_plan(line.hook, line.plan, fields, binding, between))
    bound: Final = scope or current_scope()
    if bound is not None:
        return await bound.run_plan(line.hook, line.plan_for(bound), fields, call, binding, between)
    with get_dispatcher().open_scope(_NO_SELECTION) as transient:
        return await transient.run_plan(line.hook, line.plan_for(transient), fields, call, binding, between)


def run_line_sync(
    line: HookLine,
    fields: Mapping[Field, object],
    binding: SyncLegacyBinding,
    *,
    call: CallFacts | None = None,
    scope: RequestScope | None = None,
) -> Mapping[Field, object]:
    if not line.has_natives:
        return run_legacy_plan_sync(line.plan, fields, binding)
    bound: Final = scope or current_scope()
    if bound is not None:
        return bound.run_plan_sync(line.hook, line.plan_for(bound), fields, call, binding)
    with get_dispatcher().open_scope(_NO_SELECTION) as transient:
        return transient.run_plan_sync(line.hook, line.plan_for(transient), fields, call, binding)


def kwargs_after(result: HookResult[None]) -> object:
    match result.outcome:
        case Abort() | Respond() as terminal:
            raise NativeOutcome(terminal, result.patches)
        case _:
            return result.fields[Field.deprecated_kwargs]
