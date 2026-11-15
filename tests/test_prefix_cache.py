"""Prefix cache 测试：哈希链、命中长度、最后 token 重算。"""

from leanmoe.sequence import Request, SamplingParams
from leanmoe.v1.core.block_pool import BlockPool
from leanmoe.v1.core.kv_cache import KVCacheManager


def make_request(req_id: str, tokens: list[int]) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=8),
    )


def test_prefix_hit_excludes_last_token():
    pool = BlockPool(16, enable_caching=True)
    manager = KVCacheManager(pool, block_size=16, enable_caching=True)

    tokens = list(range(40))
    first = make_request("a", tokens)
    assert manager.allocate_slots(first, num_new_tokens=40) is not None
    manager.cache_blocks(first, num_computed_tokens=40)

    second = make_request("b", tokens)
    blocks, num_hit = manager.get_computed_blocks(second)
    # 40 个 token 有 2 个满块，但最后 token 必须重算，命中上限 32
    assert num_hit == 32
    assert len(blocks) == 2


def test_identical_prefix_can_be_reused():
    pool = BlockPool(16, enable_caching=True)
    manager = KVCacheManager(pool, block_size=16, enable_caching=True)

    prompt = list(range(64))
    first = make_request("a", prompt)
    manager.allocate_slots(first, num_new_tokens=64)
    manager.cache_blocks(first, num_computed_tokens=64)

    second = make_request("b", prompt[:64])
    blocks, num_hit = manager.get_computed_blocks(second)
    assert num_hit == 48  # (64 - 1) // 16 * 16
    assert all(b.ref_cnt >= 1 for b in blocks)


def test_disabled_cache_returns_no_hit():
    pool = BlockPool(8, enable_caching=False)
    manager = KVCacheManager(pool, block_size=16, enable_caching=False)
    req = make_request("a", list(range(32)))
    blocks, num_hit = manager.get_computed_blocks(req)
    assert blocks == []
    assert num_hit == 0
