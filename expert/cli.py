import abc
import argparse
import os
from pathlib import Path

from expert_doc import get_paged_document_parser
from expert_llm import (
    GroqClient,
    JinaAiClient,
    OctoAiApiClient,
    LlmChatClient,
    LlmEmbeddingClient,
)
from expert_kb import KnowledgeBase

from expert.document_summarizer import DocumentSummarizer
from expert.kb_builder import DocumentKbBuilder
from expert.kb_interface import KbInterface


parser = argparse.ArgumentParser(
    prog="Expert Knowledge Assistant",
    description="",
)
subparsers = parser.add_subparsers(dest="command")

build_parser = subparsers.add_parser("build")
build_parser.add_argument(
    "--kb",
    dest="kb_path",
    type=str,
    required=True,
    help="Path at which to create knowledge base",
)
build_parser.add_argument(
    "--doc",
    dest="doc_path",
    type=str,
    required=True,
    help="Path to document to ingest",
)

query_parser = subparsers.add_parser("query")
query_parser.add_argument(
    "--kb",
    dest="kb_path",
    type=str,
    required=True,
    help="Path to knowledge base",
)
query_parser.add_argument(
    "--query",
    type=str,
    required=True,
    help="Query for the knowledge assistant",
)
query_parser.add_argument(
    "--n-references",
    dest="n_references",
    default=5,
    type=int,
    required=False,
    help="Number of page references to fetch",
)
query_parser.add_argument(
    "--verbose",
    action="store_true",
    required=False,
    help="Output more reference information",
)


def get_default_chat_client() -> LlmChatClient:
    return GroqClient("llama-3.1-8b-instant")


def get_default_image_client() -> LlmChatClient:
    return GroqClient("llava-v1.5-7b-4096-preview")


def get_default_embeding_client() -> LlmEmbeddingClient:
    return JinaAiClient("jina-embeddings-v2-base-en")


class Runner(abc.ABC):
    @abc.abstractmethod
    def run(self):
        return

    pass


class Runner_build(Runner):
    def __init__(
        self,
        *,
        kb_path: str,
        doc_path: str,
        index_images: bool = False,
        # TODO: actually accept string args to control these clients
        chat_client: LlmChatClient | None = None,
        embedding_client: LlmEmbeddingClient | None = None,
        **kwargs,
    ):
        self.kb_path = Path(kb_path)
        self.doc_path = Path(doc_path)
        self.chat_client = chat_client if chat_client else get_default_chat_client()
        self.embedding_client = (
            embedding_client if embedding_client else get_default_embeding_client()
        )
        self.img_client = None if not index_images else get_default_image_client()

        self.summarizer = DocumentSummarizer(
            text_client=self.chat_client,
            img_client=self.img_client,
        )
        return

    def run(self):
        builder = DocumentKbBuilder(
            embedder=self.embedding_client,
            summarizer=self.summarizer,
            path=self.doc_path,
        )
        builder.build_kb(
            dest_path=str(self.kb_path),
        )
        pass

    pass


class Runner_query(Runner):
    def __init__(
        self,
        *,
        kb_path: str,
        query: str,
        n_references: int,
        verbose: bool = False,
        # TODO: actually accept string args to control these clients
        chat_client: LlmChatClient | None = None,
        embedding_client: LlmEmbeddingClient | None = None,
        **kwargs,
    ):
        self.query = query
        self.kb_path = Path(kb_path)
        if not os.path.exists(kb_path):
            raise Exception(f"expected a knowledge base to be defined at '{kb_path}'")
        self.n_references = n_references
        self.verbose = verbose

        self.chat_client = chat_client if chat_client else get_default_chat_client()
        self.embedding_client = (
            embedding_client if embedding_client else get_default_embeding_client()
        )
        kb = KnowledgeBase(
            path=str(self.kb_path),
            embedding_size=self.embedding_client.get_embedding_vector_length(),
        )
        self.kbi = KbInterface(
            kb,
            chat_llm=self.chat_client,
            embedder=self.embedding_client,
        )
        return

    def run(self):
        res = self.kbi.chat(self.query, n_references=self.n_references)
        print(res.response)
        print("\n")
        print("REFERENCES:")

        for fragment in res.relevant_fragments:
            metadata = fragment.metadata or {}
            print("PAGE:", metadata["page"])
            if self.verbose:
                print(fragment.text)
                print()
            pass
        pass

    pass


RUNNERS = {
    "build": Runner_build,
    "query": Runner_query,
}


class Cli:
    def __init__(self):
        args = parser.parse_args()
        self.command = args.command
        if self.command not in RUNNERS:
            raise Exception(f"unknown command '{self.command}'")
        self.runner = RUNNERS[self.command](**vars(args))
        return

    def run(self):
        self.runner.run()

    pass
