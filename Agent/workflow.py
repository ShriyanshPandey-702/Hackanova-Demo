import os
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from tools.crime_classifier import classify_crime
from tools.ipc_mapper import map_ipc_section
from tools.bail_analyzer import analyze_bail_probability
from tools.summarizer import summarize_case
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def run_classifier(state: AgentState) -> dict:
    return {"crime_category": classify_crime(state["case_description"])}

def run_ipc_mapper(state: AgentState) -> dict:
    return {"ipc_sections": map_ipc_section(state["crime_category"])}

def run_bail_analyzer(state: AgentState) -> dict:
    return {"bail_probability": analyze_bail_probability(state["case_description"])}

def run_summarizer(state: AgentState) -> dict:
    return {"case_summary": summarize_case(state["case_description"])}

def synthesize_final_response(state: AgentState) -> dict:
    """Combines results and generates reasoning."""
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama-3.3-70b-versatile",
        model_kwargs={"top_p": 0.9},
        max_tokens=1200,
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    reasoning_prompt = PromptTemplate(
        input_variables=["case_description", "crime_category", "ipc_sections", "bail_probability"],
        template="""Explain the legal reasoning based on the following case facts, crime category, IPC sections, and bail probability.
Keep it professional and based on Indian legal principles.

Case facts: {case_description}
Crime Category: {crime_category}
IPC Sections: {ipc_sections}
Bail Probability: {bail_probability}

Return ONLY the reasoning text.
"""
    )
    
    reasoning_chain = reasoning_prompt | llm
    reasoning_res = reasoning_chain.invoke({
        "case_description": state["case_description"],
        "crime_category": state["crime_category"],
        "ipc_sections": state["ipc_sections"],
        "bail_probability": state["bail_probability"]
    })
    
    reasoning = reasoning_res.content.strip()
    return {"reasoning": reasoning}

def build_graph():
    graph_builder = StateGraph(AgentState)
    
    # Add nodes
    graph_builder.add_node("crime_classifier", run_classifier)
    graph_builder.add_node("ipc_mapper", run_ipc_mapper)
    graph_builder.add_node("bail_analyzer", run_bail_analyzer)
    graph_builder.add_node("case_summarizer", run_summarizer)
    graph_builder.add_node("synthesize", synthesize_final_response)
    
    # Edges
    graph_builder.add_edge(START, "crime_classifier")
    graph_builder.add_edge("crime_classifier", "ipc_mapper")
    graph_builder.add_edge("ipc_mapper", "bail_analyzer")
    graph_builder.add_edge("bail_analyzer", "case_summarizer")
    graph_builder.add_edge("case_summarizer", "synthesize")
    graph_builder.add_edge("synthesize", END)
    
    return graph_builder.compile()
