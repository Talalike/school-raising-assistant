# main_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline.rag_pipeline import run_pipeline

app = FastAPI(title="Alba Assistant API")

class CampaignRequest(BaseModel):
    school_name: str
    project_category: str
    user_input_1: str
    user_input_2: str
    user_input_3: str

# ✅ Risposta prevista
class CampaignResponse(BaseModel):
    title: str
    alt_title_1: str
    alt_title_2: str
    in_practice: str
    introduction: str
    description: str
    rewards: list

@app.post("/generate_campaign", response_model=CampaignResponse)
def generate_campaign(request: CampaignRequest):
    try:
        result = run_pipeline(
            school_name=request.school_name,
            project_category=request.project_category,
            user_input_1=request.user_input_1,
            user_input_2=request.user_input_2,
            user_input_3=request.user_input_3,
            k=5
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
