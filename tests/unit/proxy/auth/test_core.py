import dataclasses
from collections.abc import Callable
from typing import Final

import pytest

from litellm.proxy._types import (
    LiteLLM_JWTAuth,
)
from litellm.proxy.auth import core
from litellm.proxy.auth.host.python import PythonAuthHost
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.rust_bridge._native import RustAuth, RustAuthSettings, RustPlan

JWT: Final = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImsifQ.eyJzdWIiOiJ1In0.c2ln"


def _accept(claims: dict[str, object]) -> bool:
    return True


@pytest.mark.parametrize(
    ("custom_validate", "expected"),
    [(None, RustPlan.Jwt), (_accept, RustPlan.JwtWithOperatorCode)],
    ids=["no_operator_code", "custom_validate_set"],
)
def test_a_jwt_is_planned_for_operator_code_only_when_custom_validate_is_set(
    custom_validate: Callable[[dict[str, object]], bool] | None, expected: RustPlan
) -> None:
    settings: Final = RustAuthSettings(
        {"enable_jwt_auth": True},
        LiteLLM_JWTAuth().model_copy(update={"custom_validate": custom_validate}),
        core.HostFacts(master_key=None, premium=True, has_db=False),
    )
    request: Final = core.AuthRequest(
        route="/chat/completions",
        method="POST",
        token=JWT,
        model=None,
        team_header=None,
        is_llm_api_route=True,
        auth_enforced_pass_through=False,
        forwarded_routes=(),
        is_info_route=False,
        now=0.0,
    )

    assert RustAuth.plan(request, settings) == expected


@pytest.mark.asyncio
async def test_a_host_failure_the_core_cannot_skip_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch, auth_host: Callable[..., PythonAuthHost]
) -> None:
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://idp.example/jwks")
    settings: Final = RustAuthSettings(
        {"enable_jwt_auth": True},
        LiteLLM_JWTAuth(),
        core.HostFacts(master_key=None, premium=True, has_db=False),
    )

    class _Unreadable(UserApiKeyCache):
        async def async_get_cache(self, *args: object, **kwargs: object) -> object:
            raise ConnectionError("cache down")

    failing: Final = dataclasses.replace(auth_host(), user_api_key_cache=_Unreadable())
    token: Final = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImsifQ.eyJzdWIiOiJ1In0.c2ln"

    with pytest.raises(ConnectionError, match="cache down"):
        await core.drive(RustAuth.verify_jwt(token, 0.0, settings), failing)
