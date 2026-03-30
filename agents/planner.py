import json
from agents.runner import run_agent

def run_planner(user_prompt: str):

    result = run_agent(system_prompt="""You are a research planning agent. Your job is to decompose a research question 
              into a structured investigation plan.
              
              Given a research question, produce a JSON object with exactly this schema:
              {
              "sub_questions": ["string", ...],  // 3 to 6 focused, specific sub-questions that together fully cover the research question
              "keywords": [["string", ...], ...], // parallel array — 2 to 4 search keywords/phrases for each sub-question
              "scope": "string"                   // 1-2 sentences defining what this research will and will not cover
              }
              
              Rules:
              - sub_questions and keywords must be parallel arrays (keywords[i] belongs to sub_questions[i])
              - Sub-questions must be specific and answerable, not broad restatements of the original question
              - Keywords should be search-optimized phrases, not full sentences
              - scope must explicitly state boundaries — what is included AND what is out of scope
              - Respond with valid JSON only. No explanation, no markdown fences, no preamble.
              """,
              user_prompt=user_prompt
              )
    return json.loads(result)

if __name__ == "__main__":
    def test():
        result = run_planner(
            user_prompt="Explain the architecture and key innovations of the Transformer model."
        )
        print("\nFinal answer:", result)

    test()
    
