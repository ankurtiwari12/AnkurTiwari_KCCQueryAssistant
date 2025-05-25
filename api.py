from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from model import OllamaModel

app = FastAPI(title="KCC Dataset LLM API",
             description="API for interacting with the KCC dataset using LLM",
             version="1.0.0")

model = OllamaModel()

class Query(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    use_rag: bool = True
    k: int = 3

class Response(BaseModel):
    response: str
    timing: Dict[str, float]

@app.get("/")
async def root():
    return {"message": "Welcome to KCC Dataset LLM API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/generate", response_model=Response)
async def generate_response(query: Query):
    try:
        response, timing_info = model.generate_response(
            query.prompt,
            query.system_prompt,
            use_rag=query.use_rag,
            k=query.k
        )
        return Response(response=response, timing=timing_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    try:
        info = model.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 