🧠 AI Resume Analyzer

ML + NLP + LangChain + FastAPI + Streamlit

📌 Project Overview

This project is a beginner-friendly end-to-end AI system that analyzes a resume and gives structured career guidance.

The system:

Uses Machine Learning to predict the most suitable job role from resume text

Uses LangChain + LLM to generate career advice

Uses an Output Parser to keep results clean and structured

Exposes everything through an API

Displays results in a Streamlit UI

This is not a chatbot demo — it is a complete working pipeline.

🎯 What the Project Does
Input

User pastes resume text.

Output

The system returns:

Predicted job role

Missing skills

Improvement suggestions

Learning roadmap

All in a clean, structured format.

🔁 System Flow
User (Streamlit)
      ↓
FastAPI / Flask
      ↓
ML Model (Logistic Regression)
      ↓
LangChain (LLM Prompt)
      ↓
Output Parser
      ↓
API Response (JSON)
      ↓
Streamlit UI

🧩 Tech Stack
Machine Learning

Logistic Regression

TF-IDF Vectorizer

scikit-learn

NLP

TF-IDF for text vectorization

LLM Layer

LangChain

PromptTemplate

Output Parser

Backend

FastAPI (or Flask)

Frontend

Streamlit

📁 Project Structure
resume-ai/
│
├── ml/
│   ├── ml_train.py
│   ├── resume_role_model.pkl
│
├── langchain_app/
│   ├── chain.py
│   ├── parser.py
│
├── api/
│   ├── main.py
│
├── ui/
│   ├── app.py
│
├── data/
│   └── resume_sample_dataset.csv
│
├── requirements.txt
└── README.md

🧠 How the System Works (Step by Step)
1. User Input

User enters resume text in Streamlit.

2. ML Prediction

The ML model:

Converts text into numbers using TF-IDF

Uses Logistic Regression to predict a job role

Example:

Input:  "Python, Pandas, SQL, ML"
Output: "Data Scientist"

3. LangChain Reasoning

LangChain receives:

Resume text

Predicted role

It asks the LLM to generate:

Missing skills

Improvements

Learning roadmap

4. Output Parsing

The AI response is forced into this structure:

{
  "missing_skills": [],
  "improvements": [],
  "roadmap": []
}


This makes the system reliable and UI-safe.

5. API Response

FastAPI returns a clean JSON response containing:

Predicted role

Structured career advice

6. UI Display

Streamlit shows the result in a simple dashboard.

🤖 Why Logistic Regression?

This project uses Logistic Regression because:

The task is classification, not regression

It is simple, fast, and reliable

It is perfect for beginner ML pipelines

It keeps the focus on system design, not model complexity

🧪 Dataset

The dataset contains only two columns:

Column	Description
resume_text	Input text for the model
job_role	Target label

This keeps the ML task focused and understandable.

🚀 How to Run the Project
1. Install dependencies
pip install -r requirements.txt

2. Train the ML model
python ml/ml_train.py


This creates:

resume_role_model.pkl

3. Run the API
uvicorn api.main:app --reload

4. Run Streamlit UI
streamlit run ui/app.py

📌 What This Project Teaches

By building this project you learn:

How text becomes numbers using TF-IDF

How ML classification fits into a real system

How LangChain controls LLM behavior

Why Output Parsers matter

How APIs separate logic from UI

How to build a complete AI pipeline

This is system thinking, not tutorial copying.

⚠️ Limitations

This project is meant for learning, not production:

Small dataset

Simple ML model

No authentication

No database

No advanced NLP

And that is intentional.

🎓 Who This Project Is For

Beginners in ML

Students learning LangChain

Anyone who wants to understand how AI systems are built, not just how models are trained

🏁 Final Note

This project is not about building the smartest AI.
It is about building a complete working system.

If you can explain this pipeline clearly,
you have already moved ahead of most beginners.
