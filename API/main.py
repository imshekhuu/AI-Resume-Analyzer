from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated
import joblib
from pathlib import Path

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


def predict_role(resume_text: str) -> str:
    return model.predict([resume_text])[0]

@app.post("/predict", response_model=PredictionOutput)
def predict_resume(data: ResumeInput):
    role = predict_role(data.resume_text)
    return {"predicted_role": role}


@app.get("/")
def root():
    return {"message": "AI Resume Analyzer API is running"}

