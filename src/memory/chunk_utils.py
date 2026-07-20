"""文本分块工具集，用于将长文档切分为适合嵌入和检索的小块。

支持多种分块策略：固定长度、语义分割、递归分块。
"""

from __future__ import annotations

from typing import List

from src.models.schemas import SearchResult


class ChunkingStrategy:
    """分块策略基类"""

    def chunk(self, text: str) -> List[str]:
        """将文本切分为多个块"""
        raise NotImplementedError


class FixedLengthChunker(ChunkingStrategy):
    """固定长度分块策略

    将文本按固定字符数切分，简单但可能破坏语义完整性。
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        overlap = max(0, min(self.overlap, self.chunk_size - 1))
        chunks: List[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            if end >= text_len:
                break
            next_start = end - overlap
            # 保证 start 严格前进，避免 overlap 导致死循环
            start = next_start if next_start > start else end
        return chunks


class SemanticChunker(ChunkingStrategy):
    """语义分块策略

    基于句子边界和特定分隔符进行切分，尽可能保持语义完整性。
    """

    def __init__(self, max_chunk_size: int = 1024):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[str]:
        # 内部实现简化：按句子分隔，但控制每块最大长度
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # 如果当前句加上后会超出限制，先保存当前块
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = ""

            # 尝试将句子加入当前块
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # 句子太长，直接添加（不合并）
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                chunks.append(sentence)
                current_chunk = ""

        # 添加最后剩余的块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """简单句子分割"""
        import re
        # 匹配句子结束符：句号、问号、感叹号等
        sentences = re.split(r'[.!?。！？]+', text)
        return [s.strip() for s in sentences if s.strip()]


class RecursiveChunker(ChunkingStrategy):
    """递归分块策略

    基于文档结构或预先定义的分割点（如##、###标题）进行递归切分，
    适用于有结构的文档，如博客文章、Markdown文件。
    """

    def __init__(self, max_chunk_size: int = 1024):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[str]:
        # 伪实现：按##标题递归切分（简化）
        # 实际应用中应根据具体文档类型调整
        import re
        sections = re.split(r'(## [^\n]+)', text)
        if len(sections) < 2:
            # 如果没有标题，退化为语义分块
            return SemanticChunker(self.max_chunk_size).chunk(text)

        # 移除 None 和空项，并正确组织为(标题, 内容)对
        pairs = []
        i = 0
        while i < len(sections):
            if not sections[i].strip():
                i += 1
                continue
            if i + 1 < len(sections) and sections[i].startswith('## '):
                pairs.append((sections[i], sections[i+1]))
                i += 2
            else:
                # 非标题段落归属到前一个标题下
                if pairs:
                    pairs[-1] = (pairs[-1][0], pairs[-1][1] + sections[i])
                else:
                    # 首段给个默认标题
                    pairs.append(('未分段', sections[i]))
                i += 1

        chunks = []
        for title, content in pairs:
            if not content.strip():
                continue
            # 对内容部分再分块
            sub_chunks = self._chunk_content(content)
            for sub_chunk in sub_chunks:
                chunks.append(f"{title}\n{sub_chunk}")
        return chunks

    def _chunk_content(self, content: str) -> List[str]:
        """对内容再按最大长度分块"""
        chunks = []
        start = 0
        while start < len(content):
            end = min(start + self.max_chunk_size, len(content))
            chunk = content[start:end]
            chunks.append(chunk)
            start = end
        return chunks


def chunk_search_result(result: SearchResult, strategy: ChunkingStrategy) -> List[SearchResult]:
    """对单个搜索结果进行分块处理

    Args:
        result: 原始搜索结果
        strategy: 分块策略实例

    Returns:
        分块后的多个 SearchResult 列表
    """
    chunks = strategy.chunk(result.content)
    return [
        SearchResult(
            title=result.title,
            url=result.url,
            content=chunk,
            score=result.score,
        )
        for chunk in chunks
    ]


def chunk_search_results(results: List[SearchResult], strategy: ChunkingStrategy) -> List[SearchResult]:
    """对多个搜索结果并行分块处理

    Args:
        results: 搜索结果列表
        strategy: 分块策略实例

    Returns:
        合并后的所有分块结果列表
    """
    all_chunks = []
    for result in results:
        all_chunks.extend(chunk_search_result(result, strategy))
    return all_chunks