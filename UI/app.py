import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="AI Resume Analyzer")

st.title("📄 AI Resume Analyzer")

resume_text = st.text_area(
    "Paste your resume text",
    height=250
)

if st.button("Analyze Resume"):
    if not resume_text.strip():
        st.warning("Please enter resume text")
    else:
        with st.spinner("Analyzing..."):
            response = requests.post(
                API_URL,
                json={"resume_text": resume_text}
            )

            if response.status_code == 200:
                result = response.json()

                st.subheader("Predicted Role")
                st.write(result["predicted_role"])

                st.subheader("Career Advice")
                st.write(result["career_advice"])
            else:
                st.error("API error")
