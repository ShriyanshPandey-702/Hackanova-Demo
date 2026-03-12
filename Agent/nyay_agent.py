import os
import sys

# Add parent directory to path so absolute imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.workflow import build_graph

class NyayAgent:
    def __init__(self):
        # Load environment variables (like GROQ_API_KEY) from .env file
        from dotenv import load_dotenv
        load_dotenv()
            
        self.app = build_graph()
        
    def analyze_case(self, case_description: str) -> str:
        initial_state = {
            "case_description": case_description,
            "crime_category": "",
            "ipc_sections": "",
            "bail_probability": "",
            "case_summary": "",
            "reasoning": ""
        }
        
        # Run workflow
        result = self.app.invoke(initial_state)
        
        # Format final output using the results and exactly matching the specified format
        final_output = f"""Case Summary:
{result.get('case_summary', '')}

Possible IPC Sections:

{result.get('ipc_sections', '')}

Bail Probability:
{result.get('bail_probability', '')}

Reasoning:
{result.get('reasoning', '')}

Disclaimer:
This is an AI-generated legal analysis and not legal advice."""

        return final_output

if __name__ == "__main__":
    agent = NyayAgent()
    sample_case = "A female abusing an old man in public. The accused has criminal history."
    print("Running sample case analysis...")
    print("-" * 50)
    print(agent.analyze_case(sample_case))
