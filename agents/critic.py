from agents.runner import run_agent
import json

def run_critic(sub_questions: str,
               facts: str):

    result = run_agent(
        system_prompt="""You are a research critic agent. You will be given a list of sub-questions and all facts collected across those sub-questions.

        Your job is to critically evaluate the collected facts and return a JSON object with this schema:
        {
        "gaps": ["string", ...],
        "requery_needed": true | false,
        "requery_questions": ["string", ...]
        }

        Rules:
        - gaps: list every sub-question that is poorly answered, unanswered, or where facts contradict each other. Be specific about what is missing.
        - requery_needed: set to true if there are significant gaps or contradictions that would meaningfully improve the final report
        - requery_questions: if requery_needed is true, list new specific questions to fill the gaps — these should be different and more targeted than the original sub-questions. If requery_needed is false, return an empty array.
        - A gap is significant if a sub-question has fewer than 2 high-confidence facts, or if facts from different sources directly contradict each other
        - Respond with valid JSON only. No explanation, no markdown fences, no preamble.
        """,
        user_prompt=f'###Sub-question: {json.dumps(sub_questions, indent=2)} \n ###Facts: {json.dumps(facts, indent=2)}'
    )

    return json.loads(result)


if __name__ == '__main__':
    all_summaries = [{'sub_question': "What are the core components of the Transformer's encoder and decoder architecture?", 'facts': [{'claim': 'Each encoder layer contains self-attention and feed-forward neural networks, each with a residual connection and layer normalization.', 'source_url': 'https://jalammar.github.io/illustrated-transformer/', 'confidence': 'high'}, {'claim': 'Each decoder layer contains decoder self-attention, encoder-decoder attention, and a positionwise feed-forward network, each with a residual connection and layer normalization.', 'source_url': 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html', 'confidence': 'high'}, {'claim': 'Both encoder and decoder layers include feed-forward neural networks, residual connections, and layer normalization steps.', 'source_url': 'https://en.wikipedia.org/wiki/Transformer_(deep_learning)', 'confidence': 'high'}, {'claim': 'The Transformer architecture includes input embeddings, positional encoding, multi-head self-attention, feed-forward networks, and layer normalization with residual connections.', 'source_url': 'https://www.datacamp.com/tutorial/how-transformers-work', 'confidence': 'medium'}]}, {'sub_question': 'How does self-attention enable the Transformer to process sequences more effectively than RNNs?', 'facts': [{'claim': 'Self-attention enables Transformers to focus on specific parts of the input sequence that are relevant to the task, capturing long-range dependencies more effectively than traditional RNNs.', 'source_url': 'https://www.linkedin.com/pulse/understanding-self-attention-transformer-architecture-suganya-g-u9b4c', 'confidence': 'high'}, {'claim': 'Self-attention evaluates relationships within a single input sequence, allowing the model to consider the entire sequence simultaneously, unlike RNNs which process sequences step-by-step.', 'source_url': 'https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/', 'confidence': 'high'}, {'claim': 'The self-attention mechanism eliminates the need for recurrence or hidden vectors, which are required in RNNs, making the processing more efficient.', 'source_url': 'https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/', 'confidence': 'high'}]}]
    
    critic_output = run_critic(
        sub_questions=[item["sub_question"] for item in all_summaries],
        facts=[item['facts'] for item in all_summaries]
    )
    print(critic_output)