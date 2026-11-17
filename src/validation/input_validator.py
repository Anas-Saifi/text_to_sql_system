from pydantic import BaseModel, Field, field_validator


class UserQuery(BaseModel):

    question: str = Field(
        min_length=3,
        max_length=1000
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):

        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty")

        if not any(c.isalnum() for c in value):
            raise ValueError("Invalid question")

        return value