from agents.runner import run_agent
from agents.planner import run_planner
from agents.retriever import run_retriever
from agents.summarizer import run_summarizer
from agents.critic import run_critic
from agents.writer import run_writer
import json

trajectory = []

def log(agent, step, input, output, critic_loop=0):
    trajectory.append({
        "step": step,
        "agent": agent,
        "critic_loop": critic_loop,
        "input": input,
        "output": output
    })

if __name__ == '__main__':
    step = 0
    user_input = "What are the environmental and social impacts of lithium mining for electric vehicle batteries?"

    # Planner
    print('Planning...')
    planner_out = run_planner(user_input)
    log('planner', step+1, user_input, planner_out)
    step += 1

    sub_questions = planner_out['sub_questions']
    keywords = planner_out['keywords']
    scope = planner_out['scope']

    all_facts = []

    for critic_loop in range(2):

        # run retriever agent
        print('Retrieving...')
        retriever_out = run_retriever(sub_questions, keywords)
        log('retriever', step+1, sub_questions, retriever_out, critic_loop)
        step += 1

        # run summarizer agent
        print('Summarizing...')
        summarizer_out = []
        for item in retriever_out:
            summarizer_out.append(run_summarizer(item["sub_question"], item["snippets"]))
        all_facts.extend(summarizer_out)
        log('summarizer', step+1, retriever_out, summarizer_out, critic_loop)
        step += 1

        # Critic
        print('Criticing...')
        critic_out = run_critic(
            sub_questions=[item["sub_question"] for item in all_facts],
            facts=[item["facts"] for item in all_facts]
        )
        log('critic', step+1, all_facts, critic_out, critic_loop)
        step += 1

        if not critic_out['requery_needed']:
            break

        sub_questions = critic_out['requery_questions']
        keywords = [[] for _ in sub_questions]  # no keywords for requery questions

    # run writer agent
    print('Writing...')
    final_report = run_writer(
        sub_questions=[item["sub_question"] for item in all_facts],
        all_facts=[item["facts"] for item in all_facts],
        scope=scope
    )
    log('writer', step+1, all_facts, final_report)
    step += 1

    # save outputs
    out_file = "outputs/question3_report.md"
    with open(out_file, "w") as f:
        f.write(final_report)

    with open("trajectory/run_trace_question3.json", "w") as f:
        json.dump(trajectory, f, indent=2)

    print(f"Report saved to {out_file}")