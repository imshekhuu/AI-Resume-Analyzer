import joblib
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import pandas as pd
from langchain_core.prompts import PromptTemplate
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


model = joblib.load("ML\resume_role_model.pkl")