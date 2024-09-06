#!/usr/bin/env python3

from pathlib import Path

from tqdm.auto import tqdm

from expert_doc import PdfParser
from expert_llm import GroqClient, JinaAiClient
from expert_kb import KnowledgeBase, Fragment

from expert.document_summarizer import DocumentSummarizer

# 100M-Leads.pdf
# exhibit-2B-distribution-system-plan.pdf

def main():
    kb = KnowledgeBase(
        path="./navid.kb",
        embedding_size=768,
    )
    pdf_parser = PdfParser(Path("exhibit-2B-distribution-system-plan.pdf"))

    embedder = JinaAiClient("jina-embeddings-v2-base-en")
    summarizer = DocumentSummarizer(
        text_client=GroqClient("llama-3.1-8b-instant"),
        img_client=GroqClient("llava-v1.5-7b-4096-preview"),
    )

    pages = list(pdf_parser.iter_pages())
    progress_bar = tqdm(range(len(pages)))
    for i, page in enumerate(pages):
        summary = summarizer.summarize_page(page)
        texts = [
            summary.text_summary,
            *summary.img_summaries,
        ]
        embeddings = embedder.embed(texts)
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
    return


if __name__ == '__main__':
    main()
