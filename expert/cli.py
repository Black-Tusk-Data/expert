import argparse
from pathlib import Path

from expert_llm import GroqClient, JinaAiClient, OctoAiApiClient
from expert_kb import KnowledgeBase

from expert.document_summarizer import DocumentSummarizer
from expert.kb_interface import KbInterface


parser = argparse.ArgumentParser(
    prog="Expert Knowledge Assistant",
    description="",
)

parser.add_argument("--query", type=str, required=True)
parser.add_argument("--kb", type=str, required=True)


class Cli:
    def __init__(self):
        args = parser.parse_args()
        self.kb_path = Path(args.kb)
        self.query = args.query
        return

    def run(self):
        return self.run_query()

    def run_query(self):
        # TODO: don't hardcode all this stuff
        kb = KnowledgeBase(
            path=str(self.kb_path),
            embedding_size=768,
        )
        embedder = JinaAiClient("jina-embeddings-v2-base-en")
        chat_llm = OctoAiApiClient("meta-llama-3.1-8b-instruct")
        summarizer = DocumentSummarizer(
            # text_client=GroqClient("llama-3.1-8b-instant"),
            text_client=chat_llm,
            # img_client=GroqClient("llava-v1.5-7b-4096-preview"),
        )
        kbi = KbInterface(
            kb,
            chat_llm=chat_llm,
            embedder=embedder,
        )

        res = kbi.chat(self.query, n_references=5)

        print(res.response)
        print("\n")
        print("REFERENCES:")

        for fragment in res.relevant_fragments:
            metadata = fragment.metadata or {}
            print("PAGE:", metadata["page"])
            pass
        return

    pass
