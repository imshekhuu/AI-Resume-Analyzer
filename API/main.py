from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated
import joblib
from pathlib import Path
from langchain_app.llm_service import generate_career_advice

app = FastAPI(
    title="AI Resume Analyzer API",
    description="Predicts job role from resume text using ML",
    version="1.0"
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ML" / "resume_role_model.pkl"
model = joblib.load(MODEL_DIR)
# print(MODEL_DIR.exists())


class ResumeInput(BaseModel):
    resume_text: Annotated[str, Field(..., description="Enter your resume or skills")]



class PredictionOutput(BaseModel):
    predicted_role: str
    career_advice: str


def predict_role(resume_text: str) -> str:
    return model.predict([resume_text])[0]

@app.post("/predict", response_model=PredictionOutput)
def predict_resume(data: ResumeInput):
    role = predict_role(data.resume_text)

    advice = generate_career_advice(
        resume_text=data.resume_text,
        role=role
    )
    return {
        "predicted_role": role,
        "career_advice": advice
    }


@app.get("/")
def root():
    return {"message": "AI Resume Analyzer API is running"}

