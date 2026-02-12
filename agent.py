import asyncio
import os
import dotenv
from pathlib import Path

from dotenv import load_dotenv
from github import Github
from github.PullRequestReview import PullRequestReview
from llama_index.core.agent import FunctionAgent, AgentWorkflow
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult, ToolCall
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
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
    model='gpt-4o-mini',
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)


def get_changed_files(head_sha: str):
    """
        Retrieves details about files changed in the specified commit.

        Args:
            head_sha (str): The commit SHA to inspect.

        Returns:
            list[dict[str, any]]: A list of file change details including:
                - filename
                - status
                - additions
                - deletions
                - changes
                - patch (diff)
        Raises:
            ValueError: If GitHub client isn't initialized or commit isn't found.
        """
    repository = git.get_repo(full_repo_name)
    commit = repository.get_commit(head_sha)
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
    print(f"Changed files for commit {head_sha}: {changed_files}")
    return changed_files


changed_files_tool = FunctionTool.from_defaults(
    get_changed_files,
    name="get_changed_files_tool",
    description="Get the changed files in a commit by its head SHA. This is useful for understanding which files were modified in a pull request and the nature of those changes (additions, deletions, etc.).",
)


def get_file_content(file_path: str) -> str:
    """
    Retrieve the complete content of a file from the repository.

    Use this tool when you need to read the full contents of a specific file
    in the repository. This is useful for examining code context, understanding
    implementations, or reviewing files that may not have been modified in the
    current PR but are relevant to the review.

    Args:
        file_path (str): The relative path to the file in the repository
                        (e.g., 'src/main.py', 'tests/test_api.py')

    Returns:
        str: The decoded UTF-8 text content of the file.

    Raises:
        ValueError: If GitHub token is not set, the file doesn't exist,
                   or the repository is inaccessible.
    """
    repository = git.get_repo(full_repo_name)
    try:
        file_content = repository.get_contents(file_path).decoded_content.decode("utf-8")
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


async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """
    Useful for adding the gathered contexts to the state.
    """
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "State updated with report contexts. "


add_context_to_state_tool = FunctionTool.from_defaults(
    add_context_to_state,
    name="add_context_to_state_tool",
    description="Add the gathered contexts to the state. This is useful for recording the context gathered by the ContextAgent to state so that it can be used by the CommentorAgent when drafting a review comment.",
)


async def add_final_comment_to_state(ctx: Context, final_review_comment: str) -> str:
    """Useful for adding the final reviewed comment to the state."""
    current_state = await ctx.store.get("state")
    current_state["final_review_comment"] = final_review_comment
    await ctx.store.set("state", current_state)
    return "Final comment successfully added to state. "


add_final_comment_to_state_tool = FunctionTool.from_defaults(
    add_final_comment_to_state,
    name="add_final_comment_to_state_tool",
    description="Add the final review comment to the state. This is useful for recording the final review comment generated by the ReviewAndPostingAgent to state so that it can be used for posting the comment to GitHub.",
)


async def add_draft_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """
    Useful for recording the review comment generated by commentor agent to state
    """
    current_state = await ctx.store.get("state")
    current_state["review_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "Comment added to state. "


add_draft_comment_to_state_tool = FunctionTool.from_defaults(
    add_draft_comment_to_state,
    name="add_comment_to_state_tool",
    description="Add a comment to the state of the agent. This is useful for adding comments that can be used by other agents or tools in the workflow.",
)


def post_pr_comment(pr_number: int, comment: str) -> PullRequestReview:
    """
    Use this tool to get details about a specific pull request by its number. The tool will return the PR's title, author, creation date, state, and URL.
    :param comment:
    :param pr_number:
    :return:
    """
    repository = git.get_repo(full_repo_name)
    pr = repository.get_pull(pr_number)
    return pr.create_review(body=comment, event="COMMENT")


post_pr_comment_tool = FunctionTool.from_defaults(
    post_pr_comment,
    name="post_pr_comment_tool",
    description="Post a comment on a specific pull request by its number.",
)


def get_pr_details(pr_number: int) -> dict:
    """
       Use this function to retrieve details about a GitHub pull request

       Args:
           pr_number (int): The pull request number to retrieve details for.

       Returns:
           dict: A dictionary containing:
               - user (str): The GitHub username of the PR author
               - title (str): The pull request title
               - body (str): The pull request description/body text
               - state (str): Current state of the PR (e.g., 'open', 'closed')
               - diff_url (str): URL to view the PR diff
               - head_sha (str): The commit SHA of the PR's head commit

       Raises:
           ValueError: If GitHub token is not set, PR is not found, or repository
                      is inaccessible.
       """
    repository = git.get_repo(full_repo_name)
    pr = repository.get_pull(pr_number)
    commit_SHAs = []
    commits = pr.get_commits()
    for commit in commits:
        commit_SHAs.append(commit.sha)
    message_dict = {
        "author": pr.user.login,
        "title": pr.title or "missing somehow",
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "commit_SHAs": commit_SHAs
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
    description="Gathers needed context from a repository such as PR details (especially the title), changed files, file contents, and commit details. ",
    system_prompt="""
    You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
    """,
    tools=[changed_files_tool, get_pr_details_tool, get_pr_list_tool, get_file_content_tool, add_context_to_state_tool],
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
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?""",
    tools=[add_draft_comment_to_state_tool],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

review_and_poster_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Uses the tools needed to review a pull request and post a comment on GitHub. This includes using the CommentorAgent to create a review comment, running a final check to ensure the review meets the criteria, and posting the review to GitHub.",
    system_prompt="""
    You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.  
    """,
    tools=[post_pr_comment_tool, add_final_comment_to_state_tool],
    can_handoff_to=["CommentorAgent"]
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_poster_agent],
    root_agent=review_and_poster_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "fina_response": ""
    },

)


async def main():
    query = "Write a review for PR: " + pr_number
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
