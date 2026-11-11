"""A team or key in an access group that grants stores can use those stores and no others."""

import base64
from collections.abc import Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

import litellm
from litellm import Router
from litellm.models.team import LiteLLM_TeamRebuiltFromToken
from litellm.proxy._types import (
    LiteLLM_TeamTableCachedObj,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _served_by_provider_pass_through,  # pyright: ignore[reportPrivateUsage]  # the route rule has no public seam
    assert_allowed_without_access_groups,
    vector_store_access_check,
)
from litellm.proxy.auth.user_api_key_auth import _team_obj_from_token
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry
from tests.unit.proxy.vector_store_access_group_state import (
    TEAM_ID,
    TOKEN,
    ProxyState,
    make_key,
    make_team,
)


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> ProxyState:
    return ProxyState(monkeypatch)


def _virtual_key(groups: Sequence[str]) -> UserAPIKeyAuth:
    key: Final = UserAPIKeyAuth(token=TOKEN, api_key=TOKEN, team_id=TEAM_ID, access_group_ids=list(groups))
    key.via_virtual_key = True
    return key


async def _request_is_allowed(
    key: UserAPIKeyAuth, team: LiteLLM_TeamTableCachedObj | None, store: str, *, deny_by_default: bool = False
) -> bool:
    try:
        await vector_store_access_check(
            request_body={"vector_store_ids": [store]},
            team_object=team,
            valid_token=key,
            deny_by_default=deny_by_default,
        )
    except ProxyException as refused:
        if refused.code != "401":
            raise
        return False
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_stores", "key_groups", "team_stores", "team_groups", "store", "allowed"),
    [
        pytest.param(None, ("g1",), None, (), "store-a", True, id="key-group-grants"),
        pytest.param(None, ("g1",), None, (), "store-b", False, id="key-group-restricts"),
        pytest.param((), ("g1",), None, (), "store-b", False, id="key-group-restricts-with-empty-own-list"),
        pytest.param(None, (), None, ("g1",), "store-a", True, id="team-group-grants"),
        pytest.param(None, (), None, ("g1",), "store-b", False, id="team-group-restricts"),
        pytest.param(None, (), (), ("g1",), "store-b", False, id="team-group-restricts-with-empty-own-list"),
        pytest.param(("store-x",), ("g1",), None, (), "store-a", True, id="key-group-widens-own-list"),
        pytest.param(("store-x",), ("g1",), None, (), "store-x", True, id="key-own-list-still-counts"),
        pytest.param(("store-x",), ("g1",), None, (), "store-b", False, id="key-neither-list-denies"),
        pytest.param(None, (), ("store-x",), ("g1",), "store-a", True, id="team-group-widens-own-list"),
        pytest.param(None, (), ("store-x",), ("g1",), "store-x", True, id="team-own-list-still-counts"),
        pytest.param(("store-x",), (), None, ("g1",), "store-a", False, id="key-own-list-narrows-team-group"),
        pytest.param(None, ("g1",), ("store-x",), (), "store-a", False, id="team-own-list-narrows-key-group"),
        pytest.param(None, ("g1", "g2"), None, (), "store-c", True, id="stores-from-a-second-group"),
        pytest.param(None, ("models-only",), None, (), "store-b", True, id="models-only-group-changes-nothing"),
        pytest.param(None, (), None, ("models-only",), "store-b", True, id="team-models-only-group-changes-nothing"),
        pytest.param(("store-x",), ("models-only",), None, (), "store-b", False, id="models-only-keeps-own-list"),
        pytest.param(None, (), None, (), "store-b", True, id="no-grants-anywhere-changes-nothing"),
    ],
)
async def test_request_check_combines_own_list_and_group_stores(
    proxy: ProxyState,
    key_stores: Sequence[str] | None,
    key_groups: Sequence[str],
    team_stores: Sequence[str] | None,
    team_groups: Sequence[str],
    store: str,
    allowed: bool,
) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_group("g2", ("store-c",))
    await proxy.add_group("models-only", models=("gpt-4",))

    assert (
        await _request_is_allowed(make_key(key_stores, key_groups, TEAM_ID), make_team(team_stores, team_groups), store)
        is allowed
    )


@pytest.mark.asyncio
async def test_key_group_counts_when_it_lists_only_the_keys_team(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",), key_ids=())

    assert await _request_is_allowed(make_key(groups=("g1",), team_id=TEAM_ID), None, "store-a") is True
    assert await _request_is_allowed(make_key(groups=("g1",), team_id=TEAM_ID), None, "store-b") is False


@pytest.mark.asyncio
async def test_group_granted_by_an_identity_rule_counts_for_the_key(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",), team_ids=(), key_ids=())
    key: Final = make_key(("store-x",))
    key.apply_access_group_rule_grants(("g1",))

    assert await _request_is_allowed(key, None, "store-a") is True
    assert await _request_is_allowed(key, None, "store-b") is False


@pytest.mark.asyncio
async def test_key_cannot_claim_a_group_that_does_not_list_it(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",), team_ids=(), key_ids=())

    assert await _request_is_allowed(make_key(("store-x",), ("g1",)), None, "store-a") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store", "allowed"),
    [pytest.param("store-a", True, id="granted"), pytest.param("store-b", False, id="not-granted")],
)
async def test_deny_by_default_counts_group_grants(proxy: ProxyState, store: str, allowed: bool) -> None:
    await proxy.add_group("g1", ("store-a",))
    key: Final = _virtual_key(("g1",))

    assert await _request_is_allowed(key, make_team(groups=("g1",)), store, deny_by_default=True) is allowed


@pytest.mark.asyncio
async def test_deny_by_default_requires_the_key_too_not_only_its_team(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    key: Final = _virtual_key(())

    assert await _request_is_allowed(key, make_team(groups=("g1",)), "store-a", deny_by_default=True) is False


_MANAGED_A: Final = base64.urlsafe_b64encode(
    b"litellm_proxy:vector_store;unified_id,abc;target_model_names,gpt-4;resource_id,kb-shared"
).decode()
_PLAIN_B: Final = LiteLLM_ManagedVectorStore(vector_store_id="kb-shared", custom_llm_provider="bedrock")


async def _all_allowed(key: UserAPIKeyAuth, stores: Sequence[str]) -> bool:
    try:
        await vector_store_access_check(
            request_body={"vector_store_ids": list(stores)}, team_object=None, valid_token=key
        )
    except ProxyException as refused:
        if refused.code != "401":
            raise
        return False
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "plain_store", "allowed"),
    [
        pytest.param((_MANAGED_A,), None, True, id="the-listed-managed-id"),
        pytest.param(("kb-shared",), None, True, id="its-provider-id-no-store-claims"),
        pytest.param(("kb-shared",), "registry", False, id="its-provider-id-a-registered-store-claims"),
        pytest.param(("kb-shared",), "database", False, id="its-provider-id-a-database-store-claims"),
        pytest.param((_MANAGED_A, "kb-shared"), "registry", False, id="both-ids-with-a-registered-store"),
    ],
)
async def test_a_group_covers_only_the_store_the_request_runs_on(
    proxy: ProxyState,
    monkeypatch: pytest.MonkeyPatch,
    requested: Sequence[str],
    plain_store: str | None,
    allowed: bool,
) -> None:
    await proxy.add_group("g1", (_MANAGED_A,))
    registered: Final = [_PLAIN_B] if plain_store == "registry" else []
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry(vector_stores=registered))
    if plain_store == "database":
        monkeypatch.setattr(
            proxy.prisma.db.litellm_managedvectorstorestable, "find_unique", AsyncMock(return_value=MagicMock())
        )

    assert await _all_allowed(make_key(("store-own",), ("g1",)), requested) is allowed


@pytest.mark.asyncio
async def test_an_own_list_still_matches_by_provider_id(proxy: ProxyState, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry(vector_stores=[_PLAIN_B]))

    assert await _all_allowed(make_key((_MANAGED_A,)), ("kb-shared",)) is True


@pytest.mark.asyncio
async def test_a_store_lookup_that_fails_blocks_the_request(proxy: ProxyState, monkeypatch: pytest.MonkeyPatch) -> None:
    await proxy.add_group("g1", (_MANAGED_A,))
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry(vector_stores=[]))
    monkeypatch.setattr(
        proxy.prisma.db.litellm_managedvectorstorestable, "find_unique", AsyncMock(side_effect=RuntimeError("db down"))
    )

    with pytest.raises(ProxyException) as denied:
        await _all_allowed(make_key(("store-own",), ("g1",)), ("kb-shared",))

    assert denied.value.code == "503"


@pytest.mark.asyncio
async def test_a_refusal_lists_the_allowed_stores_as_main_does(proxy: ProxyState) -> None:
    with pytest.raises(ProxyException) as denied:
        await vector_store_access_check(
            request_body={"vector_store_ids": ["store-b"]}, team_object=None, valid_token=make_key(("store-x",))
        )

    assert "Only allowed to access ['store-x']" in denied.value.message


@pytest.mark.asyncio
async def test_keyless_team_member_gets_the_teams_group_stores(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    member: Final = UserAPIKeyAuth(team_id=TEAM_ID)

    assert await _request_is_allowed(member, make_team(groups=("g1",)), "store-a") is True
    assert await _request_is_allowed(member, make_team(groups=("g1",)), "store-b") is False


@pytest.mark.asyncio
async def test_deny_by_default_still_denies_a_models_only_group(proxy: ProxyState) -> None:
    await proxy.add_group("models-only", models=("gpt-4",))
    key: Final = _virtual_key(("models-only",))

    assert await _request_is_allowed(key, make_team(groups=("models-only",)), "store-a", deny_by_default=True) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_groups", "team_groups"),
    [pytest.param(("g1",), (), id="key-group"), pytest.param((), ("g1",), id="team-group")],
)
async def test_unreadable_group_blocks_the_request_instead_of_lifting_the_restriction(
    monkeypatch: pytest.MonkeyPatch, key_groups: Sequence[str], team_groups: Sequence[str]
) -> None:
    ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))

    with pytest.raises(ProxyException) as denied:
        await vector_store_access_check(
            request_body={"vector_store_ids": ["store-b"]},
            team_object=make_team(groups=team_groups),
            valid_token=make_key(groups=key_groups, team_id=TEAM_ID),
        )

    assert denied.value.code == "503"
    assert (
        denied.value.message
        == "Could not read the team, its access groups or a store list to check vector store access"
    )


@pytest.mark.asyncio
async def test_a_team_auth_rebuilt_without_groups_is_read_again_and_its_groups_still_apply(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(groups=("g1",)))
    rebuilt: Final = LiteLLM_TeamRebuiltFromToken(team_id=TEAM_ID)

    assert await _request_is_allowed(make_key(team_id=TEAM_ID), rebuilt, "store-a") is True
    assert await _request_is_allowed(make_key(team_id=TEAM_ID), rebuilt, "store-b") is False


@pytest.mark.asyncio
async def test_a_team_auth_rebuilt_without_groups_blocks_the_request_when_the_team_is_still_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ProxyState(monkeypatch, team_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))

    with pytest.raises(ProxyException) as denied:
        await vector_store_access_check(
            request_body={"vector_store_ids": ["store-b"]},
            team_object=LiteLLM_TeamRebuiltFromToken(team_id=TEAM_ID),
            valid_token=make_key(team_id=TEAM_ID),
        )

    assert denied.value.code == "503"


@pytest.mark.asyncio
async def test_a_store_list_that_cannot_be_read_blocks_a_request_its_groups_add_to(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    team: Final = make_team(groups=()).model_copy(update={"object_permission_id": "unreadable-grant"})

    with pytest.raises(ProxyException) as denied:
        await vector_store_access_check(
            request_body={"vector_store_ids": ["store-a"]}, team_object=team, valid_token=make_key((), ("g1",), TEAM_ID)
        )

    assert denied.value.code == "503"
    assert await _request_is_allowed(make_key((), (), TEAM_ID), team, "store-a") is True


@pytest.mark.asyncio
async def test_a_deleted_store_list_still_counts_as_no_list(proxy: ProxyState, monkeypatch: pytest.MonkeyPatch) -> None:
    async def deleted_row(
        object_permission_id: str,
        prisma_client: object,
        user_api_key_cache: object,
        parent_otel_span: object = None,
        proxy_logging_obj: object = None,
        check_db_only: bool = False,
    ) -> None:
        if check_db_only:
            raise HTTPException(status_code=403, detail="Referenced object permission does not exist")

    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_object_permission", deleted_row)
    await proxy.add_group("g1", ("store-a",))
    team: Final = make_team(groups=()).model_copy(update={"object_permission_id": "deleted-grant"})

    assert await _request_is_allowed(make_key((), ("g1",), TEAM_ID), team, "store-a") is True


async def _change_is_allowed(
    key: UserAPIKeyAuth, team: LiteLLM_TeamTableCachedObj | None, store: str, *, deny_by_default: bool = False
) -> bool:
    request: Final = Request({"type": "http"})
    await vector_store_access_check(
        request_body={"vector_store_ids": [store]},
        team_object=team,
        valid_token=key,
        deny_by_default=deny_by_default,
        request=request,
    )
    try:
        assert_allowed_without_access_groups(request)
    except ProxyException as refused:
        if refused.code != "401":
            raise
        return False
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_stores", "key_groups", "store", "allowed"),
    [
        pytest.param(("store-own",), ("g1",), "store-a", False, id="only-a-group-opens-it"),
        pytest.param(("store-own",), ("g1",), "store-own", True, id="own-list-allows-it"),
        pytest.param((), ("g1",), "store-a", True, id="no-own-list"),
        pytest.param((), (), "store-a", True, id="no-own-list-no-groups"),
    ],
)
async def test_a_store_change_is_refused_only_when_a_group_is_what_opens_it(
    proxy: ProxyState, key_stores: Sequence[str], key_groups: Sequence[str], store: str, allowed: bool
) -> None:
    await proxy.add_group("g1", ("store-a",))

    assert await _change_is_allowed(make_key(key_stores, key_groups), None, store) is allowed


@pytest.mark.asyncio
async def test_a_store_change_uses_the_teams_own_list_not_its_groups(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-b",))
    team: Final = make_team(("store-a",), groups=("g1",))
    await proxy.add_team(team)
    key: Final = make_key(team_id=TEAM_ID)

    assert await _change_is_allowed(key, team, "store-a") is True
    assert await _change_is_allowed(key, team, "store-b") is False


@pytest.mark.asyncio
async def test_a_store_change_needs_the_keys_own_grant_under_deny_by_default(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    group_only: Final = make_key(None, ("g1",))
    own_grant: Final = make_key(("store-a",), ("g1",))
    group_only.via_virtual_key = own_grant.via_virtual_key = True

    assert await _change_is_allowed(group_only, None, "store-a", deny_by_default=True) is False
    assert await _change_is_allowed(own_grant, None, "store-a", deny_by_default=True) is True


@pytest.mark.parametrize(
    ("module", "pass_through"),
    [
        ("litellm.proxy.pass_through_endpoints.codex_passthrough_endpoints", True),
        ("litellm.proxy.pass_through_endpoints.openai_passthrough_endpoints", True),
        ("litellm.proxy.response_api_endpoints.endpoints", False),
    ],
)
def test_every_built_in_pass_through_handler_decides_as_on_main(module: str, pass_through: bool) -> None:
    def endpoint() -> None: ...

    endpoint.__module__ = module

    assert _served_by_provider_pass_through(Request({"type": "http", "endpoint": endpoint})) is pass_through


def test_a_store_change_is_decided_as_on_main_when_the_request_check_did_not_run() -> None:
    assert_allowed_without_access_groups(Request({"type": "http"}))


@pytest.mark.asyncio
async def test_a_team_auth_rebuilt_from_the_key_keeps_the_teams_own_stores_and_groups(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(("store-own",), groups=("g1",)))
    key: Final = make_key(team_id=TEAM_ID)
    rebuilt: Final = _team_obj_from_token(key)

    assert await _request_is_allowed(key, rebuilt, "store-own") is True
    assert await _request_is_allowed(key, rebuilt, "store-a") is True
    assert await _request_is_allowed(key, rebuilt, "store-b") is False


@pytest.mark.asyncio
async def test_deleted_group_grants_nothing_and_does_not_fail_the_request(proxy: ProxyState) -> None:
    assert await _request_is_allowed(make_key(team_id=TEAM_ID), make_team(groups=("deleted",)), "store-b") is True
    assert (
        await _request_is_allowed(make_key(("store-x",), team_id=TEAM_ID), make_team(groups=("deleted",)), "store-b")
        is False
    )


@pytest.mark.asyncio
async def test_group_cached_before_the_upgrade_keeps_the_callers_own_list(proxy: ProxyState) -> None:
    await proxy.cache.async_set_cache(
        key="access_group_id:old",
        value={"access_group_id": "old", "access_group_name": "old", "assigned_team_ids": [TEAM_ID]},
    )

    assert (
        await _request_is_allowed(make_key(("store-x",), team_id=TEAM_ID), make_team(groups=("old",)), "store-b")
        is False
    )
    assert (
        await _request_is_allowed(make_key(("store-x",), team_id=TEAM_ID), make_team(groups=("old",)), "store-x")
        is True
    )


@pytest.mark.asyncio
async def test_malformed_store_ids_are_rejected_when_only_a_group_restricts(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))

    with pytest.raises(ProxyException, match="vector_store_ids must be a list of strings"):
        await vector_store_access_check(
            request_body={"model": "gpt-4o-mini", "vector_store_ids": 123},
            team_object=make_team(groups=("g1",)),
            valid_token=make_key(team_id=TEAM_ID),
            llm_router=Router(model_list=[{"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}}]),
        )


@pytest.mark.asyncio
async def test_malformed_store_ids_with_an_unreadable_group_block_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))

    with pytest.raises(ProxyException) as denied:
        await vector_store_access_check(
            request_body={"model": "gpt-4o-mini", "vector_store_ids": "store-b"},
            team_object=make_team(groups=("g1",)),
            valid_token=make_key(team_id=TEAM_ID),
            llm_router=Router(model_list=[{"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}}]),
        )

    assert denied.value.code == "503"


@pytest.mark.asyncio
async def test_request_naming_no_store_never_reads_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy: Final = ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))

    result: Final = await vector_store_access_check(
        request_body={"model": "gpt-4"}, team_object=make_team(groups=("g1",)), valid_token=make_key(team_id=TEAM_ID)
    )

    assert result is True
    assert proxy.group_reads.await_count == 0
