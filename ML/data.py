import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "resume_sample_dataset.csv"


df = pd.read_csv(DATA_DIR)

X = df["resume_text"]
y = df["job_role"]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

model = Pipeline([
    ("tfid", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy Score: ",acc)



# joblib.dump(model, "resume_role_model.pkl")
# print("Model saved as resume_role_model.pkl")



def predict_role(resume_text):
    return model.predict([resume_text])[0]

sample_text = "Python, Pandas, SQL, machine learning, data analysis"
prediction = predict_role(sample_text)
print(prediction)








