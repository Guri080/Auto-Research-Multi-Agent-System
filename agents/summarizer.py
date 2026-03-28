from runner import run_agent
import json

def run_summarizer(sub_question:str,
                   snippets: str):
    result = run_agent(
        system_prompt="""You are a research summarization agent. You will be given a sub-question and a list of text snippets retrieved from the web.

            Your job is to extract the 3 to 5 most relevant facts that answer the sub-question and return them as a JSON object with this schema:
            {
            "sub_question": "string",
            "facts": [
                {
                "claim": "string",
                "source_url": "string",
                "confidence": "high" | "medium" | "low"
                }
            ]
            }

            Rules:
            - Extract only facts that directly answer the sub-question
            - Each claim must be a single, specific, verifiable statement — not a vague summary
            - source_url must be the url of the snippet the claim came from
            - confidence scoring:
                - high: claim is stated clearly and consistently across multiple snippets
                - medium: claim is stated in one snippet with no contradiction
                - low: claim is implied, vague, or comes from a source that seems promotional or unreliable
            - Flag confidence as low if the snippet text looks like marketing copy, an opinion, or lacks specificity
            - Do not add any information not present in the snippets
            - Respond with valid JSON only. No explanation, no markdown fences, no preamble.""",
            user_prompt= f"### Sub-question: {sub_question}\n ###Snippets: {json.dumps(snippets, indent=2)}"
    )

    return json.loads(result)

if __name__ == '__main__':

    retrived_sample_text = [{'sub_question': "What are the core components of the Transformer's encoder and decoder architecture?", 'snippets': [{'title': '11.7. The Transformer Architecture — Dive into Deep Learning 1.0.3 documentation', 'text': 'Each layer is implemented in the following TransformerDecoderBlock class, which contains three sublayers:decoder self-attention, encoder–decoder attention, and positionwise feed-forward networks. These sublayers employ a residual connection around them followed by layer normalization.', 'url': 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html'}, {'title': 'The Illustrated Transformer', 'text': 'One detail in the architecture of the encoder that we need to mention before moving on, is thateach sub-layer (self-attention, ffnn) in each encoder has a residual connection around it, and is followed by a layer-normalization step. If we’re to visualize the vectors and the layer-norm operation ...', 'url': 'https://jalammar.github.io/illustrated-transformer/'}, {'title': 'Transformer (deep learning) - Wikipedia', 'text': '3 days ago -Each decoder layer contains two ... so far during inference time).Both the encoder and decoder layers have a feed-forward neural network for additional processing of their outputs and contain residual connections and layer normalization steps....', 'url': 'https://en.wikipedia.org/wiki/Transformer_(deep_learning)'}, {'title': 'How Transformers Work: A Detailed Exploration of Transformer Architecture | DataCamp', 'text': 'January 9, 2024 -Transformers are neural network ... input embeddings, positional encoding, multi-head self-attention, feed-forward networks, andlayer normalization with residual connections...', 'url': 'https://www.datacamp.com/tutorial/how-transformers-work'}, {'title': 'Architecture and Working of Transformers in Deep Learning - GeeksforGeeks', 'text': 'October 18, 2025 -Transformers are trained with teacher ... combined with multi-head attention and feed-forward networksenables highly effective handling of sequential data....', 'url': 'https://www.geeksforgeeks.org/deep-learning/architecture-and-working-of-transformers-in-deep-learning/'}]}, {'sub_question': 'How does self-attention enable the Transformer to process sequences more effectively than RNNs?', 'snippets': [{'title': 'UnderstandingSelf-AttentioninTransformerArchitecture', 'text': 'Self-attentionis a crucialmechanismintransformers; itenablesthem to focus on specific parts of the inputsequencerelevant to the task at hand and to capture long-range dependencies withinsequencesmoreeffectivelythantraditionalRNNs. WhyDoWe NeedSelf-Attention?', 'url': 'https://www.linkedin.com/pulse/understanding-self-attention-transformer-architecture-suganya-g-u9b4c'}, {'title': 'Self-AttentionMechanism- PicDictionary', 'text': 'Howdoesself-attentionimprove machine learningmodels?Self-attentionevaluates relationships within a single inputsequence, while regularattentionmechanismsmay focus on relationships between separate input and outputsequences. Why isself-attentionconsidered efficient?', 'url': 'https://picdictionary.com/ai-dictionary/self-attention-mechanism'}, {'title': 'What areTransformers? -Transformersin Artificial Intelligence...', 'text': 'Self-attentionmechanism.Theself-attentionmechanismintransformersalsoenablesthemodelto consider the entire datasequencesimultaneously. This eliminates the need for recurrence or hidden vectors.', 'url': 'https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/'}]}]

    print(retrived_sample_text[0].keys())

    all_summaries = []
    for item in retrived_sample_text:
        all_summaries.append(run_summarizer(item['sub_question'], item['snippets']))
    
    print(all_summaries)
