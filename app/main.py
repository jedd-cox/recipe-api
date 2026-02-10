import asyncio
import os
from llama_index.llms.openai import OpenAI
import dotenv
from github import Github
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult
from llama_index.core.prompts import RichPromptTemplate

dotenv.load_dotenv()
git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None

repo_url = "https://github.com/jedd-cox/recipe-api.git"
repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

print("testing prs")
if git is not None:
    repo = git.get_repo(full_repo_name)
    file_content = repo.get_contents("main.py").decoded_content.decode('utf-8')


async def main():
    query = input().strip()
    prompt = RichPromptTemplate(query)

    handler = context_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print(event.response.content)
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
