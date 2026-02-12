import asyncio
import os
import dotenv
from pathlib import Path

from dotenv import load_dotenv
from github import Github
from llama_index.core.agent import FunctionAgent, AgentWorkflow
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult, ToolCall
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# env_path = Path.home() / "Desktop" / "env" / "env"
# load_dotenv(dotenv_path=env_path)
dotenv.load_dotenv()
git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None
pr_number = os.getenv("PR_NUMBER")
repo_url = "https://github.com/jedd-cox/recipe-api.git"
repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)


def get_commit_details(sha: str):
    """
    Use this tool to get the commit details of a specific commit based on the SHA. The tool will return a list of commit details associated with the commit SHA.
    :return:
    """
    repository = git.get_repo(full_repo_name)
    commit = repository.get_commit(sha)
    changed_files: list[dict[str, any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch
        })
    return changed_files


get_commit_detail_tool = FunctionTool.from_defaults(
    get_commit_details,
    name="get_commit_details_tool",
    description="Get the commit details of a specific commit based on the SHA.",
)


def get_file_content(filename: str) -> str:
    """
    Use this tool to get the content of a specific file in the repository. The tool will return the content of the file as a string.
    :param filename:
    :return:
    """
    repository = git.get_repo(full_repo_name)
    try:
        file_content = repository.get_contents(filename).decoded_content.decode("utf-8")
        return file_content
    except Exception as e:
        return f"Error fetching file content: {str(e)}"


get_file_content_tool = FunctionTool.from_defaults(
    get_file_content,
    name="get_file_content_tool",
    description="Get the contents of a file from the repository",
)


def get_pr_list() -> list[dict[str, any]]:
    """
    Use this tool to get a list of pull requests for the repository. The tool will return a list of pull requests with their number, title, author, and state.
    :return:
    """
    repository = git.get_repo(full_repo_name)
    prs = repository.get_pulls()
    pr_list = []
    for pr in prs:
        pr_list.append({
            "number": pr.number,
            "title": pr.title,
            "author": pr.user.login,
            "state": pr.state,
        })
    return pr_list


get_pr_list_tool = FunctionTool.from_defaults(
    get_pr_list,
    name="get_pr_list_tool",
    description="Get the list of open pull requests for the repository.",
)


def add_comment_to_state(comment: str) -> str:
    """
    Use this tool to add a comment to the state of the agent. This is useful for adding comments that can be used by other agents or tools in the workflow.
    :param comment:
    :return:
    """
    return comment


add_comment_to_state_tool = FunctionTool.from_defaults(
    add_comment_to_state,
    name="add_comment_to_state_tool",
    description="Add a comment to the state of the agent. This is useful for adding comments that can be used by other agents or tools in the workflow.",
)


def get_pr_details(pr_number: int) -> dict:
    """
    Use this tool to get details about a specific pull request by its number. The tool will return the PR's title, author, creation date, state, and URL.
    :param pr_number:
    :return:
    """
    repository = git.get_repo(full_repo_name)
    pr = repository.get_pull(pr_number)
    commit_SHAs = []
    commits = pr.get_commits()
    for commit in commits:
        commit_SHAs.append(commit.sha)
    message_dict = {
        "pr_author": pr.user.login,
        "pr_title": pr.title,
        "pr_body": pr.body,
        "pr_diff_url": pr.diff_url,
        "pr_state": pr.state,
        "pr_commit_SHAs": commit_SHAs
    }
    return message_dict


get_pr_details_tool = FunctionTool.from_defaults(
    get_pr_details,
    name="get_pr_details_tool",
    description="Get the pull request details of a specific pull request by its number.",
)

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Uses the tools needed to gather all the necessary context for a pull request review. This includes the PR details, changed files, and any other relevant information from the repository. Once the context is gathered, it hands off to the Commentor Agent to write a review comment.",
    system_prompt="""
    You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
    """,
    tools=[get_commit_detail_tool, get_pr_details_tool, get_pr_list_tool, get_file_content_tool],
    can_handoff_to=["CommentorAgent"]
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="""
    You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional files, you must hand off to the Context Agent and retrieve the file.. \n
 - You should directly address the author. So your comments should sound like: \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?""",
    tools=[add_comment_to_state, get_file_content_tool],
    can_handoff_to=["ContextAgent"]
)


async def main():
    workflow_agent = AgentWorkflow(
        agents=[context_agent, commentor_agent],
        root_agent=commentor_agent.name,
        initial_state={
            "gathered_contexts": "",
            "draft_comment": "",
        }, )
    query = query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()