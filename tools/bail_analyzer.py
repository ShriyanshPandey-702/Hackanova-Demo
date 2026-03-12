import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def analyze_bail_probability(case_description: str) -> str:
    """Predicts bail likelihood based on the case facts."""
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.3-70b-versatile",
        model_kwargs={"top_p": 0.9},
        max_tokens=1200,
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    prompt = PromptTemplate(
        input_variables=["case_description"],
        template="""Predict the bail likelihood using the case facts below.
Factors to evaluate:
- violent vs non-violent crime
- criminal history
- seriousness of offense
- risk to society
- value of stolen property

Guidelines for Indian courts:
- High: Simple bailable offenses (e.g., verbal abuse/insult, simple assault without injuries), even if the accused has a prior history, provided the current offense is very minor and non-violent. Verbal altercations and abuse with NO criminal history should always be High.
- Medium: Non-bailable but less heinous crimes (e.g., mobile phone theft, simple cheating). Criminal history may increase bail difficulty but doesn't guarantee a "Low" rating if the current offense is minor.
- Low: Heinous crimes (murder, rape, dacoity, severe fraud), offenses with severe violence, or if the current offense is serious AND the accused has a significant criminal history or flight risk.

Case facts:
{case_description}

Output ONLY one of the following words: Low, Medium, High
"""
    )
    
    chain = prompt | llm
    response = chain.invoke({"case_description": case_description})
    output = response.content.strip()
    if "High" in output:
        return "High"
    elif "Medium" in output:
        return "Medium"
    elif "Low" in output:
        return "Low"
    return output
