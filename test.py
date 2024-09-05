#!/usr/bin/env python3

from pathlib import Path

from expert_doc import PdfParser
from expert_llm import GroqClient

from expert.document_summarizer import DocumentSummarizer


def main():
    pdf_parser = PdfParser(Path("./100M-Leads.pdf"))
    pages = list(
        pdf_parser.iter_pages()
    )
    
    summarizer = DocumentSummarizer(
        text_client=GroqClient("llama-3.1-8b-instant"),
        img_client=GroqClient("llava-v1.5-7b-4096-preview"),
    )

    from IPython import embed; embed()
    return


if __name__ == '__main__':
    main()

