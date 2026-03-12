import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def map_ipc_section(crime_category: str) -> str:
    """Maps a crime category to possible IPC sections."""
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.3-70b-versatile",
        model_kwargs={"top_p": 0.9},
        max_tokens=1200,
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    prompt = PromptTemplate(
        input_variables=["crime_category"],
        template="""Map the following crime category to relevant Indian Penal Code (IPC) sections.
Provide the output as a list in this exact format:
* Section <number> - <description>
* Section <number> - <description>

Example mappings:
Theft → IPC 379
Assault → IPC 351 / 352
Cheating → IPC 420
Murder → IPC 302
Criminal intimidation → IPC 506

Crime Category:
{crime_category}
"""
    )
    
    chain = prompt | llm
    response = chain.invoke({"crime_category": crime_category})
    return response.content.strip()
