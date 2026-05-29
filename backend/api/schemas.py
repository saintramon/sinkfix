from pydantic import BaseModel

class ModelTextInput(BaseModel):
    model_name: str
    text: str

class ModelResponse(BaseModel):
    token_list: list[str]
    corrected_att_scores: list[list[list[list[float]]]]
    classifications: list[str]