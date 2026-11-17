from pydantic import BaseModel, Field

class InputValidator(BaseModel):
    valid_input: bool = Field(description = "True if input is valid otherwise False")
    user_input: str = Field(description = "User input if it is valid")
    incomplete_request: bool = Field(description = "True if user input is incomplete, vague or not understandable otherwise False")
    invalid_reason: str = Field(description = "reason for why the input is invalid if it is invalid otherwise null")

class ResultValidator(BaseModel):
    valid_result: bool = Field(description = "True if result is correct otherwise False")
    correction: str = Field(description = "Feedback for the sql executor to generate correct prompt if it is incorrect otherwise null")
    corrected_sql: str = Field(description = "the correct sql if the input sql is wrong")