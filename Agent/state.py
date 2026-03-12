from typing import TypedDict

class AgentState(TypedDict):
    case_description: str
    crime_category: str
    ipc_sections: str
    bail_probability: str
    case_summary: str
    reasoning: str
