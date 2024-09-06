from pydantic import BaseModel, Field

from expert_kb import Fragment


class KbChatResponse(BaseModel):
    relevant_fragments: list[Fragment] = Field(default_factory=list)
    response: str
    pass
