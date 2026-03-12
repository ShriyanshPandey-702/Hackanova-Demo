import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def classify_crime(case_description: str) -> str:
    """Classifies the crime based on the case description."""
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.3-70b-versatile",
        model_kwargs={"top_p": 0.9},
        max_tokens=1200,
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    prompt = PromptTemplate(
        input_variables=["case_description"],
        template="""Analyze the following case description and classify it into a single primary crime category (e.g., Theft, Assault, Fraud, Murder, Cybercrime, Criminal intimidation, etc.).
Return ONLY the crime category name, nothing else.

Case description:
{case_description}
"""
    )
    
    chain = prompt | llm
    response = chain.invoke({"case_description": case_description})
    return response.content.strip()
