from expert_kb import KnowledgeBase
from expert_llm.models import ChatBlock, LlmChatClient, LlmEmbeddingClient

from expert.models import KbChatResponse


class KbInterface:
    def __init__(
            self,
            # TODO: Should be some way of constructing the embedder
            #       that corresponds to the KB.
            knowledge_base: KnowledgeBase,
            *,
            chat_llm: LlmChatClient,
            embedder: LlmEmbeddingClient,
    ):
        self.kb = knowledge_base
        self.chat_llm = chat_llm
        self.embedder = embedder
        return

    def chat(
            self,
            prompt: str,
            *,
            n_references: int = 5,
    ) -> KbChatResponse:
        prompt_embedding = self.embedder.embed([prompt])[0]
        corpus_fragments = self.kb.search(
            prompt_embedding,
            k=n_references,
        )
        text_response = self.chat_llm.chat_completion([
            ChatBlock(
                role="system",
                content="\n".join([
                    (
                        "You are a helpful knowledge assistant."
                        "  Your responses must ONLY reference the following context:"
                    ),
                    "---- BEGIN CONTEXT ----",
                    *[
                        fragment.text
                        for fragment in corpus_fragments
                    ],
                    "---- END CONTEXT ----",
                ]),
            ),
            ChatBlock(
                role="user",
                content=prompt,
            ),
        ])
        return KbChatResponse(
            response=text_response.content,
            relevant_fragments=corpus_fragments,
        )

    pass
