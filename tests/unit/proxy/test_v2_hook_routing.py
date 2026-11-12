import asyncio
from collections.abc import Awaitable, Callable, Generator, Sequence
from typing import Final

import pytest
from fastapi import HTTPException
from mcp.types import CallToolResult, TextContent

import litellm
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.exceptions import RejectedRequestError, SensitiveDataRouteException
from litellm.integrations.custom_logger_v2 import (
    Abort,
    CallContext,
    CustomLoggerV2,
    Field,
    Hook,
    Patch,
    build_manifest,
    on,
)
from litellm.integrations.custom_logger_v2.host import NativeOutcome, register_native, reset_host_for_tests
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.policy_engine.pipeline_types import GuardrailPipeline, PipelineStep
from litellm.types.utils import CallTypesLiteral, GenericGuardrailAPIInputs, ModelResponse

AUTH: Final = UserAPIKeyAuth(api_key="sk-test")


@pytest.fixture(autouse=True)
def fresh_host() -> Generator[None]:
    reset_host_for_tests()
    yield
    reset_host_for_tests()


@pytest.fixture
def proxy_logging() -> ProxyLogging:
    return ProxyLogging(user_api_key_cache=DualCache())


class _Witness(CustomLoggerV2):
    def __init__(self, order: list[str]) -> None:
        self.order = order

    @on(Hook.pre_call, may_abort=True)
    async def pre_call(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.during_call, may_abort=True)
    async def during_call(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.post_call_success, may_abort=True)
    async def post_call_success(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.post_call_failure, may_abort=True)
    async def post_call_failure(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.response_headers, writes={Field.response_headers})
    async def response_headers(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.list_models, may_abort=True)
    async def list_models(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.mcp_tool_result, may_abort=True)
    async def mcp_tool_result(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")


class _Legacy(CustomLogger):
    def __init__(self, order: list[str], reply: object = None, error: Exception | None = None) -> None:
        super().__init__()
        self.order = order
        self.reply = reply
        self.error = error
        self.seen: list[object] = []

    def _record(self, seen: object) -> object:
        self.order.append("legacy")
        self.seen.append(seen)
        if self.error is not None:
            raise self.error
        return self.reply

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: CallTypesLiteral
    ) -> object:
        return self._record(data)

    async def async_moderation_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, call_type: CallTypesLiteral
    ) -> object:
        return self._record(data)

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        return self._record(response)

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> object:
        return self._record(original_exception)

    async def async_post_call_response_headers_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
        request_headers: dict[str, str] | None = None,
        litellm_call_info: dict[str, object] | None = None,
    ) -> object:
        return self._record(response)

    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        self._record(tuple(model_names))
        return ("kept",)


class _McpGuardrail(CustomGuardrail):
    def __init__(self, order: list[str]) -> None:
        super().__init__(guardrail_name="mcp-witness", default_on=True, event_hook=GuardrailEventHooks.post_mcp_call)
        self.order = order

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: str,
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        self.order.append("legacy")
        return inputs


class _StreamGuardrail(CustomGuardrail):
    def __init__(self, order: list[str], reply: object) -> None:
        super().__init__(guardrail_name="stream-witness", default_on=True, event_hook=GuardrailEventHooks.post_call)
        self.order = order
        self.reply = reply

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        self.order.append("legacy")
        return self.reply


class _SuccessDispatch:
    def __init__(self) -> None:
        self.model_call_details: dict[str, object] = {"metadata": {}}
        self.dispatched: list[object] = []

    async def dispatch_success_handlers(self, result: object, **kwargs: object) -> None:
        self.dispatched.append(result)


async def test_pre_call_passthrough_hands_the_request_to_legacy_once_and_returns_its_reply(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    reply: Final[dict[str, object]] = {"model": "replaced", "metadata": {}}
    legacy: Final = _Legacy(order, reply=reply)
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    data: Final[dict[str, object]] = {"model": "gpt-4o", "metadata": {}}

    result: Final = await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert result is reply
    assert legacy.seen == [data]
    assert legacy.seen[0] is data


async def test_pre_call_passthrough_propagates_the_legacy_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    error: Final = ValueError("rejected by a legacy hook")
    monkeypatch.setattr(litellm, "callbacks", [_Legacy([], error=error)])

    with pytest.raises(ValueError, match="rejected by a legacy hook") as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "gpt-4o"}, call_type="completion")

    assert raised.value is error


async def test_during_call_passthrough_returns_the_request_and_propagates_the_legacy_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    legacy: Final = _Legacy(order)
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    data: Final = {"model": "gpt-4o"}

    result: Final = await proxy_logging.during_call_hook(data=data, user_api_key_dict=AUTH, call_type="completion")

    assert result is data
    assert legacy.seen == [data]
    error: Final = RuntimeError("moderation failed")
    monkeypatch.setattr(litellm, "callbacks", [_Legacy([], error=error)])
    with pytest.raises(RuntimeError) as raised:
        await proxy_logging.during_call_hook(data=data, user_api_key_dict=AUTH, call_type="completion")
    assert raised.value is error


async def test_post_call_success_passthrough_returns_the_legacy_replacement(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    replacement: Final = ModelResponse()
    legacy: Final = _Legacy([], reply=replacement)
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    response: Final = ModelResponse()

    result: Final = await proxy_logging.post_call_success_hook(
        data={"model": "gpt-4o"}, response=response, user_api_key_dict=AUTH
    )

    assert result is replacement
    assert legacy.seen == [response]
    assert legacy.seen[0] is response


async def test_post_call_success_passthrough_propagates_the_legacy_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    error: Final = ValueError("response rejected")
    monkeypatch.setattr(litellm, "callbacks", [_Legacy([], error=error)])

    with pytest.raises(ValueError, match="response rejected") as raised:
        await proxy_logging.post_call_success_hook(
            data={"model": "gpt-4o"}, response=ModelResponse(), user_api_key_dict=AUTH
        )

    assert raised.value is error


async def test_post_call_success_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_Legacy(order)])

    await proxy_logging.post_call_success_hook(
        data={"model": "gpt-4o"}, response=ModelResponse(), user_api_key_dict=AUTH
    )

    assert order == ["native", "legacy"]


async def test_post_call_failure_passthrough_returns_the_first_legacy_http_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    transformed: Final = HTTPException(status_code=418, detail="rewritten")
    legacy: Final = _Legacy([], reply=transformed)
    monkeypatch.setattr(litellm, "callbacks", [legacy, _Legacy([], reply=HTTPException(status_code=500))])
    original: Final = RuntimeError("upstream failed")

    result: Final = await proxy_logging.post_call_failure_hook(
        request_data={"model": "gpt-4o"}, original_exception=original, user_api_key_dict=AUTH
    )

    assert result is transformed
    assert legacy.seen == [original]
    assert legacy.seen[0] is original


async def test_post_call_failure_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_Legacy(order)])

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "gpt-4o"}, original_exception=RuntimeError("failed"), user_api_key_dict=AUTH
    )

    assert order == ["native", "legacy"]


async def test_response_headers_passthrough_returns_the_legacy_headers(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    legacy: Final = _Legacy([], reply={"x-legacy": "1"})
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    response: Final = ModelResponse()

    result: Final = await proxy_logging.post_call_response_headers_hook(
        data={"model": "gpt-4o"}, user_api_key_dict=AUTH, response=response
    )

    assert result == {"x-legacy": "1"}
    assert legacy.seen == [response]


async def test_response_headers_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_Legacy(order, reply={"x-legacy": "1"})])

    await proxy_logging.post_call_response_headers_hook(
        data={"model": "gpt-4o"}, user_api_key_dict=AUTH, response=ModelResponse()
    )

    assert order == ["native", "legacy"]


async def test_list_models_passthrough_returns_the_names_legacy_hides(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    legacy: Final = _Legacy([])
    monkeypatch.setattr(litellm, "callbacks", [legacy])

    result: Final = await proxy_logging.hidden_by_listing_callbacks(AUTH, ["kept", "hidden"])

    assert result == frozenset({"hidden"})
    assert legacy.seen == [("kept", "hidden")]


async def test_list_models_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_Legacy(order)])

    await proxy_logging.hidden_by_listing_callbacks(AUTH, ["kept"])

    assert order == ["native", "legacy"]


async def test_post_mcp_call_passthrough_returns_the_result_untouched_without_guardrails(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    response: Final = CallToolResult(content=[TextContent(type="text", text="tool output")])

    result: Final = await proxy_logging.post_mcp_call_hook(response=response, request_data={}, user_api_key_dict=AUTH)

    assert result is response


async def test_post_mcp_call_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_McpGuardrail(order)])
    response: Final = CallToolResult(content=[TextContent(type="text", text="tool output")])

    await proxy_logging.post_mcp_call_hook(response=response, request_data={"metadata": {}}, user_api_key_dict=AUTH)

    assert order == ["native", "legacy"]


async def test_deferred_stream_guardrails_passthrough_dispatches_the_guardrail_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: Final[list[str]] = []
    replacement: Final = ModelResponse()
    monkeypatch.setattr(litellm, "callbacks", [_StreamGuardrail(order, replacement)])
    dispatch: Final = _SuccessDispatch()

    await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
        captured_data={"model": "gpt-4o", "metadata": {}},
        captured_user_api_key_dict=AUTH,
        captured_logging_obj=dispatch,
        assembled_response=ModelResponse(),
        cache_hit=None,
    )
    await _drain()

    assert order == ["legacy"]
    assert dispatch.dispatched == [replacement]
    assert dispatch.dispatched[0] is replacement


async def test_deferred_stream_guardrails_native_runs_before_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_StreamGuardrail(order, None)])

    await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
        captured_data={"model": "gpt-4o", "metadata": {}},
        captured_user_api_key_dict=AUTH,
        captured_logging_obj=_SuccessDispatch(),
        assembled_response=ModelResponse(),
        cache_hit=None,
    )

    assert order == ["native", "legacy"]


async def _drain() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


Action = Callable[[dict[str, object]], Awaitable[None] | Exception | str | dict[str, object] | None]


def _passes(data: dict[str, object]) -> None:
    return None


class _PreCall(CustomLogger):
    def __init__(self, name: str, journal: list[object], action: Action = _passes, enforces: bool = False) -> None:
        super().__init__()
        self.name = name
        self.journal = journal
        self.action = action
        self.enforces_request_content = enforces
        self.seen: list[dict[str, object]] = []

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: CallTypesLiteral
    ) -> Exception | str | dict[str, object] | None:
        self.journal.append(self.name)
        self.seen.append(data)
        return await _settled(self.action(data))

    async def async_moderation_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, call_type: CallTypesLiteral
    ) -> object:
        self.journal.append(self.name)
        await asyncio.sleep(0)
        return await _settled(self.action(data))


class _Guard(CustomGuardrail):
    def __init__(
        self,
        name: str,
        journal: list[object],
        action: Action = _passes,
        parallel: bool = False,
        delay: float = 0.0,
        event: GuardrailEventHooks = GuardrailEventHooks.pre_call,
    ) -> None:
        super().__init__(guardrail_name=name, default_on=True, event_hook=event, run_in_parallel=parallel)
        self.name = name
        self.journal = journal
        self.action = action
        self.delay = delay
        self.seen_models: list[object] = []

    async def _run(self, data: dict[str, object]) -> Exception | str | dict[str, object] | None:
        self.journal.append(self.name)
        self.seen_models.append(data.get("model"))
        await asyncio.sleep(self.delay)
        return await _settled(self.action(data))

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: CallTypesLiteral
    ) -> Exception | str | dict[str, object] | None:
        return await self._run(data)

    async def async_moderation_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, call_type: CallTypesLiteral
    ) -> object:
        return await self._run(data)


class _Native(CustomLoggerV2):
    def __init__(self, journal: list[object], replacement: object = None, outcome: Abort | None = None) -> None:
        self.journal = journal
        self.replacement = replacement
        self.outcome = outcome

    @on(Hook.pre_call, reads={Field.messages}, writes={Field.messages}, may_abort=True)
    async def pre_call(self, view: object, ctx: CallContext[object]) -> Patch | Abort | None:
        self.journal.append(("native", getattr(view, "messages", None)))
        if self.outcome is not None:
            return self.outcome
        return None if self.replacement is None else Patch().set(Field.messages, self.replacement)

    @on(Hook.during_call, reads={Field.messages}, may_abort=True)
    async def during_call(self, view: object, ctx: CallContext[object]) -> Abort | None:
        self.journal.append("native")
        await asyncio.sleep(0)
        return self.outcome


async def _settled(
    value: Awaitable[None] | Exception | str | dict[str, object] | None,
) -> Exception | str | dict[str, object] | None:
    return await value if isinstance(value, Awaitable) else value


def _reroute(name: str) -> Action:
    def action(data: dict[str, object]) -> None:
        raise SensitiveDataRouteException(
            route_to_model=name, session_id="s", guardrail_name=name, sticky_session_routing=False
        )

    return action


def _raise(error: BaseException) -> Action:
    def action(data: dict[str, object]) -> None:
        raise error

    return action


def _returning(value: str | dict[str, object] | None) -> Action:
    def action(data: dict[str, object]) -> str | dict[str, object] | None:
        return value

    return action


def _with_inert_native(callbacks: list[object], inert: bool) -> list[object]:
    return [*callbacks, _Native([])] if inert else callbacks


@pytest.mark.parametrize("inert", [False, True])
async def test_pre_call_runs_sequential_entries_in_list_order_then_the_parallel_guardrails(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    callbacks: Final[list[object]] = [
        _Guard("parallel-1", journal, parallel=True),
        _PreCall("logger-1", journal),
        _Guard("sequential", journal),
        _Guard("parallel-2", journal, parallel=True),
        _PreCall("logger-2", journal),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert_native(callbacks, inert))

    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")

    assert journal == ["logger-1", "sequential", "logger-2", "parallel-1", "parallel-2"]


@pytest.mark.parametrize("inert", [False, True])
async def test_pre_call_dict_replaces_the_request_and_none_keeps_it(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    replacement: Final[dict[str, object]] = {"model": "replaced"}
    first: Final = _PreCall("first", [], _returning(replacement))
    keeper: Final = _PreCall("keeper", [])
    last: Final = _PreCall("last", [])
    monkeypatch.setattr(litellm, "callbacks", _with_inert_native([first, keeper, last], inert))
    data: Final[dict[str, object]] = {"model": "m"}

    result: Final = await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert first.seen[0] is data
    assert keeper.seen[0] is replacement
    assert last.seen[0] is replacement
    assert result is replacement


@pytest.mark.parametrize(("call_type", "error"), [("completion", RejectedRequestError), ("embeddings", HTTPException)])
async def test_pre_call_string_rejects_and_stops_the_line(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, call_type: CallTypesLiteral, error: type[Exception]
) -> None:
    journal: Final[list[object]] = []
    monkeypatch.setattr(
        litellm, "callbacks", [_PreCall("rejecter", journal, _returning("no thanks")), _PreCall("after", journal)]
    )

    with pytest.raises(error):
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type=call_type)

    assert journal == ["rejecter"]


async def test_pre_call_sequential_exception_propagates_unchanged_and_stops_the_line(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    error: Final = HTTPException(status_code=403, detail="blocked")
    monkeypatch.setattr(litellm, "callbacks", [_Guard("blocker", journal, _raise(error)), _PreCall("after", journal)])

    with pytest.raises(HTTPException) as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")

    assert raised.value is error
    assert journal == ["blocker"]


@pytest.mark.parametrize("inert", [False, True])
async def test_parallel_guardrails_pick_the_first_block_in_list_order_and_a_block_beats_a_reroute(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    slow_error: Final = ValueError("slow but first")
    callbacks: Final[list[object]] = [
        _Guard("reroute", journal, _reroute("onprem"), parallel=True),
        _Guard("slow", journal, _raise(slow_error), parallel=True, delay=0.02),
        _Guard("fast", journal, _raise(HTTPException(status_code=400)), parallel=True),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert_native(callbacks, inert))
    metadata: Final[dict[str, object]] = {}
    data: Final[dict[str, object]] = {"model": "m", "metadata": metadata}

    with pytest.raises(ValueError, match="slow but first") as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert raised.value is slow_error
    assert data["model"] == "m"
    assert metadata["applied_guardrails"] == ["slow", "fast"]


async def test_parallel_reroute_alone_reroutes_the_request(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    monkeypatch.setattr(litellm, "callbacks", [_Guard("reroute", journal, _reroute("onprem"), parallel=True)])

    result: Final = await proxy_logging.pre_call_hook(
        user_api_key_dict=AUTH, data={"model": "m", "metadata": {}}, call_type="completion"
    )

    assert result["model"] == "onprem"


@pytest.mark.parametrize("inert", [False, True])
async def test_deferred_reroute_lets_later_guardrails_run_and_the_first_reroute_wins(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    later: Final = _Guard("later", journal)
    parallel: Final = _Guard("parallel", journal, parallel=True)
    callbacks: Final[list[object]] = [
        _Guard("first-reroute", journal, _reroute("first")),
        _Guard("second-reroute", journal, _reroute("second")),
        later,
        parallel,
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert_native(callbacks, inert))

    result: Final = await proxy_logging.pre_call_hook(
        user_api_key_dict=AUTH, data={"model": "m", "metadata": {}}, call_type="completion"
    )

    assert journal == ["first-reroute", "second-reroute", "later", "parallel"]
    assert later.seen_models == ["m"]
    assert parallel.seen_models == ["first"]
    assert result["model"] == "first"
    assert result["metadata"]["sensitive_data_routing_original_model"] == "m"


async def test_a_block_after_a_deferred_reroute_still_propagates(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    error: Final = HTTPException(status_code=400, detail="blocked")
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [_Guard("reroute", [], _reroute("onprem")), _Guard("blocker", [], _raise(error))],
    )
    data: Final[dict[str, object]] = {"model": "m", "metadata": {}}

    with pytest.raises(HTTPException) as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert raised.value is error
    assert data["model"] == "m"


async def test_a_pipeline_managed_guardrail_runs_in_its_pipeline_and_not_again_in_the_line(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    monkeypatch.setattr(litellm, "callbacks", [_Guard("managed", journal), _Guard("unmanaged", journal)])
    pipeline: Final = GuardrailPipeline(mode="pre_call", steps=[PipelineStep(guardrail="managed", on_pass="next")])
    data: Final[dict[str, object]] = {
        "model": "m",
        "messages": [],
        "metadata": {"_guardrail_pipelines": [("p", pipeline)]},
    }

    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert journal == ["managed", "unmanaged"]


@pytest.mark.parametrize(
    ("skip_guardrails", "guardrails_only", "expected"),
    [
        (False, False, ["logger", "guard", ("native", None), "enforcer"]),
        (True, False, ["logger", ("native", None), "enforcer"]),
        (False, True, ["guard", "enforcer"]),
    ],
)
async def test_call_site_modes_choose_which_entries_run(
    monkeypatch: pytest.MonkeyPatch,
    proxy_logging: ProxyLogging,
    skip_guardrails: bool,
    guardrails_only: bool,
    expected: list[object],
) -> None:
    journal: Final[list[object]] = []
    callbacks: Final[list[object]] = [
        _PreCall("logger", journal),
        _Guard("guard", journal),
        _Native(journal),
        _PreCall("enforcer", journal, enforces=True),
    ]
    monkeypatch.setattr(litellm, "callbacks", callbacks)

    await proxy_logging.pre_call_hook(
        user_api_key_dict=AUTH,
        data={"model": "m"},
        call_type="completion",
        skip_guardrails=skip_guardrails,
        guardrails_only=guardrails_only,
    )

    assert journal == expected


async def test_a_runtime_change_to_the_callback_list_reaches_the_next_request(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    first: Final = _PreCall("first", journal)
    callbacks: Final[list[object]] = [first]
    monkeypatch.setattr(litellm, "callbacks", callbacks)

    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")
    callbacks.append(_PreCall("added", journal))
    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")
    callbacks.remove(first)
    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")

    assert journal == ["first", "first", "added", "added"]


async def test_a_native_between_two_legacy_loggers_sees_the_first_edit_and_the_second_sees_its_patch(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []

    def edit(data: dict[str, object]) -> None:
        data["messages"] = ["edited by the first"]
        return None

    seen_by_second: Final[list[object]] = []

    def record(data: dict[str, object]) -> None:
        seen_by_second.append(data["messages"])

    first: Final = _PreCall("first", [], edit)
    second: Final = _PreCall("second", [], record)
    monkeypatch.setattr(litellm, "callbacks", [first, _Native(journal, replacement=["patched by the native"]), second])
    data: Final[dict[str, object]] = {"model": "m", "messages": ["original"]}

    result: Final = await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert journal == [("native", ["edited by the first"])]
    assert seen_by_second == [["patched by the native"]]
    assert result is data
    assert data["messages"] == ["patched by the native"]


async def test_a_native_abort_stops_the_line_before_later_legacy_loggers(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    abort: Final = Abort(status=429, message="slow down")
    monkeypatch.setattr(
        litellm, "callbacks", [_PreCall("before", journal), _Native(journal, outcome=abort), _PreCall("after", journal)]
    )

    with pytest.raises(NativeOutcome) as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")

    assert raised.value.outcome is abort
    assert journal == ["before", ("native", None)]


@pytest.mark.parametrize("inert", [False, True])
async def test_during_call_starts_every_entry_in_list_order_and_returns_on_the_first_failure(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    release: Final = asyncio.Event()
    error: Final = RuntimeError("moderation failed")

    async def hold(data: dict[str, object]) -> None:
        await release.wait()

    slow: Final = _Guard("slow", journal, hold, event=GuardrailEventHooks.during_call)
    callbacks: Final[list[object]] = [
        slow,
        _PreCall("moderator", journal, _raise(error)),
        _Guard("guard", journal, event=GuardrailEventHooks.during_call),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert_native(callbacks, inert))

    with pytest.raises(RuntimeError) as raised:
        await asyncio.wait_for(
            proxy_logging.during_call_hook(data={"model": "m"}, user_api_key_dict=AUTH, call_type="completion"), 1
        )

    release.set()
    await _drain()
    assert raised.value is error
    assert journal[:3] == ["slow", "moderator", "guard"]


async def test_during_call_native_between_legacy_runs_with_them_and_its_abort_wins_by_list_order(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    abort: Final = Abort(status=400, message="flagged")
    callbacks: Final[list[object]] = [
        _PreCall("first", journal),
        _Native(journal, outcome=abort),
        _PreCall("second", journal),
    ]
    monkeypatch.setattr(litellm, "callbacks", callbacks)

    with pytest.raises(NativeOutcome) as raised:
        await proxy_logging.during_call_hook(data={"model": "m"}, user_api_key_dict=AUTH, call_type="completion")

    assert raised.value.outcome is abort
    assert journal == ["first", "native", "second"]
