from fastAPI import APIRouter
from schemas import ModelResponse, ModelTextInput

router = APIRouter()

@router.post("/analyze")
async def analyze():
