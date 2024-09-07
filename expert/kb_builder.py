import abc
import os
from pathlib import Path

from tqdm.auto import tqdm

from expert_doc import get_paged_document_parser
from expert_llm import LlmEmbeddingClient, LlmChatClient
from expert_kb import KnowledgeBase

from expert.document_summarizer import DocumentSummarizer


class KbBuilder(abc.ABC):
    def __init__(
            self,
            *,
            embedder: LlmEmbeddingClient,
            summarizer: DocumentSummarizer,
    ) -> None:
        self.embedder = embedder
        self.summarizer = summarizer
        return

    @abc.abstractmethod
    def build_kb(
            self,
            dest_path: str,
            *,
            force_fresh: bool = False,
    ) -> KnowledgeBase:
        pass

    pass


class DocumentKbBuilder(KbBuilder):
    def __init__(
            self,
            *,
            embedder: LlmEmbeddingClient,
            summarizer: DocumentSummarizer,
            path: Path,
    ) -> None:
        super().__init__(embedder=embedder, summarizer=summarizer)
        self.path = path
        return

    def build_kb(
            self,
            dest_path: str,
            *,
            force_fresh: bool = False,
    ) -> KnowledgeBase:
        if force_fresh:
            if os.path.exists(dest_path):
                os.remove(dest_path)
                pass
            pass
        parser = get_paged_document_parser(self.path)
        kb = KnowledgeBase(
            path=dest_path,
            embedding_size=self.embedder.get_embedding_vector_length(),
        )

        last_ingested_page = kb.db.query("""
        SELECT max(metadata_json->>'page') last_page
          FROM embedded_fragment
        """)[0]["last_page"]
        # page numbers are '1' indexed...
        last_ingested_page_idx = (last_ingested_page or 0) - 1

        pages = list(parser.iter_pages())
        progress_bar = tqdm(range(len(pages)))
        for i, page in enumerate(pages):
            if i <= last_ingested_page_idx:
                progress_bar.update(1)
                continue
            summary = self.summarizer.summarize_page(page)
            texts = [
                summary.text_summary,
                *summary.img_summaries,
            ]
            embeddings = self.embedder.embed(texts)
            kb.add_fragment(
                fragment_id=f"page-{i + 1}",
                text=texts[0],
                embedding=embeddings[0],
                metadata={"page": i + 1},
            )
            for j in range(1, len(texts)):
                text = texts[j]
                embed = embeddings[j]
                kb.add_fragment(
                    fragment_id=f"page-{i + 1}-image-{j}",
                    text=text,
                    embedding=embed,
                    metadata={"page": i + 1},
                )
                pass
            progress_bar.update(1)
            pass
        return kb

    pass
