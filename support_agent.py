# support_agent.py
import os
import json
import requests

from langchain.llms import OpenAI
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool
from langchain.prompts import PromptTemplate  # Use PromptTemplate instead

# ---------------------------------------------------------------------
# Tool Functions: These functions call the TicketFlow API.
# ---------------------------------------------------------------------
TICKETFLOW_API_URL = os.environ.get("TICKETFLOW_API_URL", "http://localhost:8000")


def create_ticket(params: str) -> str:
    """
    Create a new support ticket.
    Expected params: JSON string with keys "description" and "user".
    """
    try:
        data = json.loads(params)
        description = data.get("description")
        user = data.get("user")
        if not description or not user:
            return "Error: Both 'description' and 'user' must be provided."
    except Exception as e:
        return f"Error parsing parameters: {str(e)}"

    payload = {"description": description, "user": user}
    response = requests.post(f"{TICKETFLOW_API_URL}/tickets", json=payload)
    if response.status_code == 200:
        ticket = response.json()
        return f"Ticket created with ID: {ticket['id']}"
    else:
        return f"Failed to create ticket: {response.text}"


def update_ticket(params: str) -> str:
    """
    Update an existing support ticket.
    Expected params: JSON string with keys "ticket_id", and optionally "description" and/or "status".
    """
    try:
        data = json.loads(params)
        ticket_id = data.get("ticket_id")
        description = data.get("description")
        status = data.get("status")
        if ticket_id is None or (description is None and status is None):
            return "Error: 'ticket_id' and at least one of 'description' or 'status' must be provided."
    except Exception as e:
        return f"Error parsing parameters: {str(e)}"

    payload = {}
    if description is not None:
        payload["description"] = description
    if status is not None:
        payload["status"] = status
    response = requests.put(f"{TICKETFLOW_API_URL}/tickets/{ticket_id}", json=payload)
    if response.status_code == 200:
        return f"Ticket {ticket_id} updated successfully."
    else:
        return f"Failed to update ticket: {response.text}"


def query_ticket(params: str) -> str:
    """
    Query a support ticket.
    Expected params: A string representing the ticket_id.
    """
    try:
        ticket_id = int(params.strip())
    except Exception as e:
        return f"Error parsing ticket_id: {str(e)}"

    response = requests.get(f"{TICKETFLOW_API_URL}/tickets/{ticket_id}")
    if response.status_code == 200:
        ticket = response.json()
        return f"Ticket {ticket_id}: {json.dumps(ticket)}"
    else:
        return f"Failed to query ticket: {response.text}"


# ---------------------------------------------------------------------
# Wrap our functions as LangChain Tools.
# ---------------------------------------------------------------------
tools = [
    Tool(
        name="Create Ticket",
        func=create_ticket,
        description=(
            "Use this tool to create a support ticket. "
            "Input should be a JSON string containing 'description' and 'user'."
        )
    ),
    Tool(
        name="Update Ticket",
        func=update_ticket,
        description=(
            "Use this tool to update a support ticket. "
            "Input should be a JSON string containing 'ticket_id' and optionally 'description' and/or 'status'."
        )
    ),
    Tool(
        name="Query Ticket",
        func=query_ticket,
        description=(
            "Use this tool to query a support ticket. "
            "Input should be a ticket ID (an integer) provided as a string."
        )
    )
]

# ---------------------------------------------------------------------
# Define a System Prompt with Few-shot Examples
# ---------------------------------------------------------------------
prefix = (
    "You are a customer support assistant that uses the TicketFlow API to create, update, "
    "and query support tickets for users. Your goal is to interpret user requests and decide "
    "which tool to use. Always follow the format provided in the examples."
)

suffix = (
    "\n\nWhen you receive a question, think carefully about the necessary action and provide "
    "your answer in the following format:\n"
    "Thought: <your reasoning>\n"
    "Action: <the tool to use>\n"
    "Action Input: <input for the tool>\n"
    "Observation: <result of the action>\n"
    "Final Answer: <final answer to the user>\n\n"
    "Question: {input}\n"
)

examples = [
    """Input: I cannot log in to my account.
Thought: The user is experiencing a login issue and needs a ticket created.
Action: Create Ticket
Action Input: {"description": "I cannot log in to my account", "user": "unknown"}
Observation: Ticket created with ID: 1
Final Answer: I have created a support ticket for your login issue. Ticket ID is 1.""",
    """Input: Please update ticket 1 to change its status to closed.
Thought: The user wants to update an existing ticket.
Action: Update Ticket
Action Input: {"ticket_id": 1, "status": "closed"}
Observation: Ticket 1 updated successfully.
Final Answer: Ticket 1 has been updated and marked as closed.""",
    """Input: What are the details of ticket 1?
Thought: The user is asking for details of a ticket.
Action: Query Ticket
Action Input: "1"
Observation: Ticket 1: {"id": 1, "description": "I cannot log in to my account", "user": "unknown", "status": "open"}
Final Answer: Here are the details of ticket 1: {"id": 1, "description": "I cannot log in to my account", "user": "unknown", "status": "open"}."""
]

# Manually build the prompt template string.
prompt_template_str = (
        f"{prefix}\n\n" +
        "\n\n".join(examples) +
        "\n\n" +
        suffix
)

prompt_template = PromptTemplate(
    input_variables=["input"],
    template=prompt_template_str,
)

# ---------------------------------------------------------------------
# Initialize the LLM and the Zero-shot Agent.
# ---------------------------------------------------------------------
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

agent = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt_template)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# ---------------------------------------------------------------------
# Main loop: interactively receive user commands.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Welcome to the SupportAgent. Type your command (or 'exit' to quit):")
    while True:
        user_input = input("Command: ")
        if user_input.lower().strip() == "exit":
            break
        result = agent_executor.run(user_input)
        print(result)
