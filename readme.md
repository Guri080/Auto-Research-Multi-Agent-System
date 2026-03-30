# AutoResearch

A multi-agent system that takes a research question and returns a structured report. You give it a question, it searches the web, pulls out the relevant facts, checks its own work, and writes everything up with citations.

Built for CSE 475. Uses `gpt-4.1-mini` by default.

---

## Setup

```bash
git clone <repo>
cd autoresearch
python -m venv myENV
source myENV/bin/activate
pip install -r requirements.txt
```

Add a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

Set your question in `main.py` and run:

```bash
python main.py
```

Output goes to `outputs/report.md`. The full agent trajectory is saved to `outputs/trajectory.json`.

---

## Architecture


Every agent talks to the next one through JSON. No shared state. The full conversation each agent had with the model is logged to `trajectory.json`.

---

## Agent Design Decisions

**Planner only runs once.** I thought about having the Planner rerun if the Critic found big gaps, but that felt like overkill. The Critic already produces targeted follow-up questions which is basically replanning anyway, just cheaper.

**No tool-calling in the Retriever.** The DuckDuckGo search is just a plain Python function call. The LLM only sees the results and formats them into JSON. I tried tool-calling first and it added a lot of complexity for no real benefit since I knew exactly which API I was hitting.

**Critic looks at everything at once.** This was important to get right. If you run the Critic per sub-question it can not catch cases where two sub-questions returned conflicting information. Running it on all facts together fixes that.

**Writer is not allowed to add new information.** The system prompt is pretty strict about this. It can only use facts that came from the Summarizer. This was the simplest way I could think of to reduce hallucination in the final report without doing anything fancy.

**Critic loop is capped at 2.** Mostly a cost decision. In practice the Critic rarely asked for a second requery on these questions anyway.

---

## Limitations

- DuckDuckGo only returns short snippets, not full pages. For some questions the facts end up pretty surface-level because there just was not enough text to work with.

- The model sometimes returns its JSON wrapped in markdown fences even though the prompt says not to. There is a helper that strips those out but occasionally the JSON is broken in other ways and the run just crashes. Re-running usually fixes it.

- The Critic sometimes says everything is fine when it clearly is not. It depends a lot on how detailed the facts were going in.

- If you run several questions in a row DuckDuckGo will start rate limiting you and return empty results. Just wait a bit.

---

## Cost

I ran 3 questions and spent about $0.05 total using `gpt-4.1-mini`.

Questions tested:
1. "What are the primary mechanisms by which large language models hallucinate, and what mitigation strategies have been proposed?"
2. "Summarise the history and current state of quantum error correction."
3. "What are the environmental and social impacts of lithium mining for electric vehicle batteries?"

---

## Project Structure

```
autoresearch/
├── agents/
│   ├── __init__.py
│   ├── planner.py
│   ├── retriever.py
│   ├── summarizer.py
│   ├── critic.py
│   └── writer.py
├── retrieval/
│   ├── __init__.py
│   └── duckduckgo.py
├── outputs/
│   ├── report.md
│   └── trajectory.json
├── runner.py
├── main.py
├── requirements.txt
└── .env
```