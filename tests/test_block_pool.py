"""块池行为测试：分配、释放、LIFO/FIFO、缓存命中。"""

from leanmoe.utils import sha256_hash
from leanmoe.v1.core.block_pool import BlockPool


def test_allocate_and_free_returns_blocks():
    pool = BlockPool(4, enable_caching=False)
    blocks = pool.get_new_blocks(2)
    assert [b.block_id for b in blocks] == [0, 1]
    assert all(b.ref_cnt == 1 for b in blocks)
    assert pool.num_free_blocks == 2

    pool.free_blocks(blocks)
    assert pool.num_free_blocks == 4
    # 无哈希块回队首，下次分配优先复用
    again = pool.get_new_blocks(2)
    assert {b.block_id for b in again} == {0, 1}


def test_cached_block_goes_to_tail():
    pool = BlockPool(4, enable_caching=True)
    first = pool.get_new_blocks(1)[0]
    h = sha256_hash(b"a")
    pool.cache_full_blocks([first], [h], num_cached_blocks=0, num_full_blocks=1)

    pool.free_blocks([first])
    # 有哈希块在队尾；分配应先拿到队首的未缓存块
    nxt = pool.get_new_blocks(1)[0]
    assert nxt.block_id != first.block_id
    assert first.block_hash == h


def test_cache_hit_increases_ref_and_removes_from_free_queue():
    pool = BlockPool(2, enable_caching=True)
    block = pool.get_new_blocks(1)[0]
    h = sha256_hash(b"b")
    pool.cache_full_blocks([block], [h], num_cached_blocks=0, num_full_blocks=1)
    pool.free_blocks([block])
    assert pool.num_free_blocks == 2

    hit = pool.get_cached_block(h)
    assert hit is block
    assert hit.ref_cnt == 1
    assert pool.num_free_blocks == 1


def test_free_queue_order_is_fifo_for_cached_blocks():
    pool = BlockPool(3, enable_caching=True)
    blocks = pool.get_new_blocks(2)
    hashes = [sha256_hash(b"x"), sha256_hash(b"y")]
    pool.cache_full_blocks(blocks, hashes, 0, 2)
    pool.free_blocks(blocks)
    # 0/1 有哈希，释放后进队尾；从未分配的 2 号块仍在队首
    a = pool.get_new_blocks(1)[0]
    assert a.block_id == 2
    # 再取一块应拿到有哈希的 0 或 1（队尾顺序），且哈希被清除
    b = pool.get_new_blocks(1)[0]
    assert b.block_id in (0, 1)
    assert b.block_hash is None
