import base64
from pathlib import Path
from typing import NamedTuple

from expert_doc import Image, ParsedPage
from expert_llm import (
    LlmChatClient,
    LlmEmbeddingClient,
    ChatBlock,
)


TMP_DIR = Path("/tmp")


class PageSummary(NamedTuple):
    text_summary: str
    img_summaries: list[str]
    pass


class DocumentSummarizer:
    def __init__(
        self,
        *,
        text_client: LlmChatClient,
        img_client: LlmChatClient | None = None,
    ):
        self.text_client = text_client
        self.img_client = img_client
        return

    def summarize_page(self, page: ParsedPage) -> PageSummary:

        text_prompt = self._get_text_summarization_prompt(page)
        text_completion = self.text_client.chat_completion(text_prompt)
        text_summary = text_completion.content

        img_summaries: list[str] = []
        if self.img_client:
            for image in page.images:
                img_prompt = self._get_img_summarization_prompt(image)
                img_completion = self.img_client.chat_completion(img_prompt)
                img_summaries.append(img_completion.content)
                pass

        return PageSummary(
            text_summary=text_summary,
            img_summaries=img_summaries,
        )

    def _get_img_summarization_prompt(self, image: Image) -> list[ChatBlock]:
        # The Groq llava model does not support a system prompt, which is why we only use a user prompt here.
        # Indeed, we might want to have any clients of the DocumentSummarizer specify the prompt format.
        blocks = []
        img_fname = image.dump_to_file(str(TMP_DIR / ".doc-image"))
        with open(img_fname, "rb") as f:
            blocks.append(
                ChatBlock(
                    role="user",
                    content=f"Summarize the information contained in the following image:",
                    image_b64=base64.b64encode(f.read()).decode("utf-8"),
                )
            )
        return blocks

    def _get_text_summarization_prompt(self, page: ParsedPage) -> list[ChatBlock]:
        system_prompt = " ".join(
            [
                "You are a helpful expert in a huge number of topics.",
                "You are deisgned to summarize each page of a technical document, one at a time.",
                "Given the contents of a single page of a document, you respond with a SUCCINCT summary of the information on that page.",
            ]
        )
        blocks = [
            ChatBlock(
                role="system",
                content=system_prompt,
            )
        ]
        blocks.append(
            ChatBlock(
                role="user",
                content="\n".join(
                    [
                        "PAGE CONTENTS:",
                        page.text,
                    ]
                ),
            )
        )
        return blocks

    pass
