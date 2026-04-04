from typing import List

from pydantic import BaseModel


class CustomResetRequest(BaseModel):
    seed: int
    difficulty: str
    custom_description: str
    custom_logs: List[str]
