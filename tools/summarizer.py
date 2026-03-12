import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def summarize_case(case_description: str) -> str:
    """Generates a brief summary of the case."""
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.3-70b-versatile",
        model_kwargs={"top_p": 0.9},
        max_tokens=1200,
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    prompt = PromptTemplate(
        input_variables=["case_description"],
        template="""Generate a concise summary of the following criminal case description.
Length must strictly be 2-3 sentences. Do not add any extra commentary.

Case description:
{case_description}
"""
    )
    
    chain = prompt | llm
    response = chain.invoke({"case_description": case_description})
    return response.content.strip()
