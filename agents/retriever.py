import json
from ddgs import DDGS

from agents.runner import run_agent


def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    results = list(DDGS().text(query, max_results=5))

    return results  # returns title, href, and body

def run_retriever(sub_questions: list,
                 keyword: list):

    all_results = []
    for i, question in enumerate(sub_questions):
        query = question + " " + " ".join(keyword[i])

        search_result = search_duckduckgo(query)
        result = run_agent(
            system_prompt="""You are a retrieval agent. You will be given a sub-question and raw search results from DuckDuckGo.
            
                            Your job is to select the most relevant results and return them as a JSON object with this schema:
                            {
                            "sub_question": "string",
                            "snippets": [
                                {
                                "title": "string",
                                "text": "string",
                                "url": "string"
                                }
                            ]
                            }

                            Rules:
                            - Include only snippets that are directly relevant to the sub-question
                            - Keep up to 5 snippets maximum
                            - Use the body field as text, href field as url, title field as title
                            - Do not summarize or modify the text — copy it as-is from the search result
                            - Respond with valid JSON only. No explanation, no markdown fences, no preamble.
            """,
            user_prompt=f"Sub-question: {question}\n\nRaw search results:\n{json.dumps(search_result, indent=2)}"
        )

        all_results.append(json.loads(result))

    return all_results


if __name__ =='__main__':
    result = run_retriever(
            sub_questions=[ 
                "What are the core components of the Transformer's encoder and decoder architecture?",
                "How does self-attention enable the Transformer to process sequences more effectively than RNNs?"
                ],
            keyword=[
                ["Transformer encoder architecture", "Transformer decoder architecture", "multi-head attention mechanism", "residual connections in Transformer"],
                ["self-attention mechanism", "sequence modeling with attention", "attention weights in Transformers", "how self-attention works"]
            ]
        )
    
    print(result)