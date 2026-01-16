import joblib
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import pandas as pd
from langchain_core.prompts import PromptTemplate
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).parent
ML_DIR = BASE_DIR / 'ML' / 'resume_role_model.pkl'
model = joblib.load(ML_DIR)
# print(ML_DIR.exists())

st.sidebar.title("📄 AI Resume Analyzer")
resume_text = st.sidebar.text_area("enter your resume text")
# print(response)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation"
)

llm_model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    input_variables=["resume_text", "role"],
    template="""
You are a professional career coach.

The job role has already been predicted by a machine learning model.
Do NOT guess the role yourself.

Predicted Job Role:
{role}

Resume:
{resume_text}

Provide your response under these headings only:

Missing Skills:
- 

Improvements:
- 

Learning Roadmap:
- 
"""
)

if st.sidebar.button("Analyze Resume"):
    if not resume_text.strip():
            st.warning("Please enter resume text")
    else:
        role = model.predict([resume_text])[0]


    final_prompt = prompt.format(
        resume_text = resume_text,
        role = role
    )

    respone = llm_model.invoke(final_prompt)
    st.subheader("Predicted Role")
    st.write(role)

    st.subheader("Career Advice")
    st.write(respone.content)


