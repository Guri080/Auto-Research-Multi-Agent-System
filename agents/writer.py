from runner import run_agent

def run_writer(
        sub_questions:str, 
        all_facts: str,
        scope: str
):

    result = run_agent(
        system_prompt="""
        You are a research writing agent. You will be given a research scope, a list of sub-questions, and all verified facts collected across those sub-questions.

        Your job is to synthesise the facts into a structured Markdown research report.

        The report must follow this exact structure:

        # [Report Title]

        ## Executive Summary
        A 3-5 sentence overview of the key findings across all sub-questions.

        ## [Section per sub-question]
        One section for each sub-question, using the sub-question as the section heading.
        Synthesise the relevant facts into coherent paragraphs. Cite sources inline as [Source](url).

        ## Limitations
        A honest assessment of what this report could not cover, where sources were thin, and what a reader should verify independently.

        ## References
        A numbered list of all source URLs cited in the report.

        Rules:
        - Do not add any information not present in the provided facts
        - Do not fabricate citations — only cite URLs that appear in the facts
        - Every claim in the body must be traceable to at least one fact
        - Write in clear, neutral, academic prose
        - The limitations section must mention any sub-questions that had low-confidence facts or sparse coverage
        - Return only the Markdown string. No JSON, no preamble, no explanation.
        """,
        user_prompt=f"## sub-questions: {sub_questions}\n ## facts: {all_facts} \n ## Scope: {scope}"
    )

    return result

if __name__ == '__main__':

    input_text = [{'sub_question': "What are the core components of the Transformer's encoder and decoder architecture?", 'facts': [{'claim': 'Each encoder layer contains self-attention and feed-forward neural networks, each with a residual connection and layer normalization.', 'source_url': 'https://jalammar.github.io/illustrated-transformer/', 'confidence': 'high'}, {'claim': 'Each decoder layer contains decoder self-attention, encoder-decoder attention, and a positionwise feed-forward network, each with a residual connection and layer normalization.', 'source_url': 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html', 'confidence': 'high'}, {'claim': 'Both encoder and decoder layers include feed-forward neural networks, residual connections, and layer normalization steps.', 'source_url': 'https://en.wikipedia.org/wiki/Transformer_(deep_learning)', 'confidence': 'high'}, {'claim': 'The Transformer architecture includes input embeddings, positional encoding, multi-head self-attention, feed-forward networks, and layer normalization with residual connections.', 'source_url': 'https://www.datacamp.com/tutorial/how-transformers-work', 'confidence': 'medium'}]}, {'sub_question': 'How does self-attention enable the Transformer to process sequences more effectively than RNNs?', 'facts': [{'claim': 'Self-attention enables Transformers to focus on specific parts of the input sequence that are relevant to the task, capturing long-range dependencies more effectively than traditional RNNs.', 'source_url': 'https://www.linkedin.com/pulse/understanding-self-attention-transformer-architecture-suganya-g-u9b4c', 'confidence': 'high'}, {'claim': 'Self-attention evaluates relationships within a single input sequence, allowing the model to consider the entire sequence simultaneously, unlike RNNs which process sequences step-by-step.', 'source_url': 'https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/', 'confidence': 'high'}, {'claim': 'The self-attention mechanism eliminates the need for recurrence or hidden vectors, which are required in RNNs, making the processing more efficient.', 'source_url': 'https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/', 'confidence': 'high'}]}]
    scope =  {"scope": "This research covers the structural design and key technical innovations of the original Transformer model as introduced in 'Attention Is All You Need'. It focuses on architectural components, attention mechanisms, and training enhancements. It does not cover later variants (e.g., BERT, GPT), applications in specific domains, or implementation details in frameworks like PyTorch or TensorFlow."}
    

    result = run_writer(
        sub_questions=[item['sub_question'] for item in input_text],
        all_facts=[item['facts'] for item in input_text],
        scope=scope['scope']
    )

    print(result)