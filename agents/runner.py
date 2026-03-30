import os
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv()) 


MAX_ITERATIONS = 2
# MODEL = 'qwen3-30b-a3b-instruct-2507'
MODEL = 'gpt-4.1-mini'


def run_agent(
        system_prompt: str,
        user_prompt: str,
        tools: list = None,   
        tool_executer=None 
) -> str:
    
    client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
    )

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f'[runnner] iteration {iteration + 1}')

        kwargs = {'model': MODEL,
                  "messages": messages}
        
        if tools:
            kwargs["tools"] = tools

        response = client.chat.completions.create(**kwargs)

        response_message = response.choices[0].message

        if not response_message.tool_calls:
            print("[runner] model gave final answer")
            return response_message.content
        
        print(f"[runner] model wants {len(response_message.tool_calls)} tool calls")
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            name = tool_call.function.name

            args = json.loads(tool_call.function.arguments)
            print(f'[runner] calling tool: {name}({args})')

            if tool_executer:
                result = tool_executer(name, args)
            else:
                result = f'[no executor registered for {name}]'

            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': str(result)
            })
        
    print('[runner] hit max iterations without final answer')
    return ''

## method for testing file
if __name__ == '__main__':

    def test():
        result = run_agent(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello in 1 line"
        )
        print("\nFinal answer:", result)

    test()



