"""分块工具单元测试。"""

from src.memory.chunk_utils import FixedLengthChunker, SemanticChunker, chunk_search_result
from src.models.schemas import SearchResult


def test_fixed_length_chunker_with_overlap_terminates():
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    chunker = FixedLengthChunker(chunk_size=20, overlap=5)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert "".join(c[0] for c in chunks)  # non-empty
    # 不应死循环：块数有上界
    assert len(chunks) < len(text)


def test_fixed_length_chunker_overlap_ge_size_safe():
    chunker = FixedLengthChunker(chunk_size=10, overlap=10)
    chunks = chunker.chunk("0123456789ABCDEF")
    assert chunks
    assert len(chunks) >= 2


def test_semantic_chunker_splits_sentences():
    text = "第一句。第二句！第三句？结尾"
    chunks = SemanticChunker(max_chunk_size=8).chunk(text)
    assert len(chunks) >= 2


def test_chunk_search_result():
    result = SearchResult(
        title="t",
        url="http://example.com",
        content="A" * 100,
        score=0.9,
    )
    chunks = chunk_search_result(result, FixedLengthChunker(chunk_size=30, overlap=0))
    assert len(chunks) >= 3
    assert all(c.url == result.url for c in chunks)
