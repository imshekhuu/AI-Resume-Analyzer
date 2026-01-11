from langchain_core.prompts import PromptTemplate

def get_prompt():
    prompt = PromptTemplate(
        template="""You are a professional career coach.

The system has already predicted the user's job role using machine learning.
Do NOT guess the role yourself.

Predicted Job Role:
{role}

Resume Text:
{resume_text}

Your task:
Analyze the resume strictly for the given role and return the response in clear bullet points.

Return the output in the following format only:

Missing Skills:
- 

Improvements:
- 

Learning Roadmap:
- 
""",
input_variables=['role', 'resume_text']
)
    
    return prompt