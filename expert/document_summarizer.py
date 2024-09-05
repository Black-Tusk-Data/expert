import base64
from pathlib import Path

from expert_doc import Image, ParsedPage
from expert_llm import (
    LlmChatClient,
    LlmEmbeddingClient,
    ChatBlock,
)


TMP_DIR = Path("/tmp")


class DocumentSummarizer:
    def __init__(
            self,
            *,
            text_client: LlmChatClient,
            img_client: LlmChatClient,
    ):
        self.text_client = text_client
        self.img_client = img_client
        return

    def summarize_page(self, page: ParsedPage) -> list[str]:
        text_prompt = self._get_text_summarization_prompt(page)
        text_completion = self.text_client.chat_completion(text_prompt)
        summaries = [text_completion.content]

        for image in page.images:
            img_prompt = self._get_img_summarization_prompt(image)
            img_completion = self.img_client.chat_completion(img_prompt)
            summaries.append(img_completion.content)
            pass
        
        return summaries

    def _get_img_summarization_prompt(self, image: Image) -> list[ChatBlock]:
        system_prompt = " ".join([
            "You are a helpful expert in a huge number of topics.",
            "You are deisgned to summarize images from pages of a technical document, one page at a time.",
            "Given an image from a single page of a document, you respond with a SUCCINCT summary of the information described in that images.",
        ])
        blocks = [ChatBlock(
            role="system",
            content=system_prompt,
        )]
        img_fname = image.dump_to_file(str(TMP_DIR / ".doc-image"))
        with open(img_fname, "rb") as f:
            blocks.append(ChatBlock(
                role="user",
                content=f"Summarize the following image:",
                image_b64=base64.b64encode(f.read()).decode("utf-8"),
            ))
        return blocks


    def _get_text_summarization_prompt(self, page: ParsedPage) -> list[ChatBlock]:
        system_prompt = " ".join([
            "You are a helpful expert in a huge number of topics.",
            "You are deisgned to summarize each page of a technical document, one at a time.",
            "Given the contents of a single page of a document, you respond with a SUCCINCT summary of the information on that page.",
        ])
        blocks = [ChatBlock(
            role="system",
            content=system_prompt,
        )]
        blocks.append(ChatBlock(
            role="user",
            content="\n".join([
                "PAGE CONTENTS:",
                page.text,
            ])
        ))
        return blocks

    pass
