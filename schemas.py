from pydantic import BaseModel, Field

class MessageCreate(BaseModel):
    customer: str = Field(min_length=1)
    message: str = Field(min_length=1)

