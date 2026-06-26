# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo
import os

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import Workflow, START, node
from google.genai import types
from pydantic import BaseModel

# Safe auth setup
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() in ("true", "1", "yes")

# If GEMINI_API_KEY is provided, we can default to AI Studio unless vertex is explicitly set to True
if os.environ.get("GEMINI_API_KEY") and "GOOGLE_GENAI_USE_VERTEXAI" not in os.environ:
    use_vertex = False

if use_vertex:
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    try:
        _, project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get("GOOGLE_CLOUD_PROJECT", project_id)
        os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    except DefaultCredentialsError:
        # Fallback to AI Studio if credentials are not found but an API key is available
        if os.environ.get("GEMINI_API_KEY"):
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
        else:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
else:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


def get_weather(query: str) -> str:
    """Simulates a web search. Use it get information on weather.

    Args:
        query: A string containing the location to get weather information for.

    Returns:
        A string with the simulated weather information for the queried location.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


def get_current_time(query: str) -> str:
    """Simulates getting the current time for a city.

    Args:
        city: The name of the city to get the current time for.

    Returns:
        A string with the current time information.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return f"Sorry, I don't have timezone information for query: {query}."

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return f"The current time for query {query} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"


# Define the output schema for input classification
class Classification(BaseModel):
    is_expense: bool
    amount: float = 0.0
    description: str = ""


# Create LLM-based classifier agent node
classifier = LlmAgent(
    name="classifier",
    model="gemini-flash-latest",
    mode="single_turn",
    instruction=(
        "Classify the user prompt. Check if they want to submit, request approval, "
        "or log an expense (e.g. 'submit expense of $50 for lunch', 'approve travel expense $120'). "
        "If so, extract the amount and description, and set is_expense to True. "
        "If it is a general question, greeting, or weather query, set is_expense to False. "
        "Always output exactly in the required schema."
    ),
    output_schema=Classification,
)


# Define routing function node
@node
def router(node_input: dict) -> Event:
    """Routes the workflow based on whether it is an expense or a general query."""
    if node_input.get("is_expense"):
        return Event(output=node_input, route="expense")
    return Event(output=node_input, route="general")


# Define human-in-the-loop expense approval node
@node
async def approve_expense(ctx: Context, node_input: dict):
    """Prompts for human approval using RequestInput, and handles response."""
    if not ctx.resume_inputs:
        amount = node_input.get("amount", 0.0)
        description = node_input.get("description", "unspecified item")
        yield RequestInput(
            interrupt_id="expense_approval",
            message=f"I detected a request to log an expense of ${amount:.2f} for '{description}'. Do you want to approve this expense? (yes/no)"
        )
        return

    approval = ctx.resume_inputs.get("expense_approval", "")
    if str(approval).lower().strip() in ("yes", "approve", "approved", "y"):
        yield Event(output=node_input, route="approved")
    else:
        yield Event(output=node_input, route="rejected")


# Define approval terminal nodes
@node
def save_expense(node_input: dict):
    """Handles approved expenses."""
    amount = node_input.get("amount", 0.0)
    description = node_input.get("description", "unspecified item")
    message = f"Expense of ${amount:.2f} for '{description}' has been approved and saved successfully."
    yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=message)]))
    yield Event(output=message)


@node
def reject_expense(node_input: dict):
    """Handles rejected expenses."""
    amount = node_input.get("amount", 0.0)
    description = node_input.get("description", "unspecified item")
    message = f"Expense of ${amount:.2f} for '{description}' has been rejected."
    yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=message)]))
    yield Event(output=message)


# Define the general assistant LLM agent node
general_assistant = LlmAgent(
    name="general_assistant",
    model="gemini-flash-latest",
    mode="single_turn",
    instruction=(
        "You are a helpful AI assistant designed to provide accurate and useful information. "
        "Use the weather or current time tools if needed to answer the user's query."
    ),
    tools=[get_weather, get_current_time],
)


# Instantiate the root workflow agent
root_agent = Workflow(
    name="root_agent",
    edges=[
        (START, classifier),
        (classifier, router),
        (router, {"expense": approve_expense, "general": general_assistant}),
        (approve_expense, {"approved": save_expense, "rejected": reject_expense}),
    ],
)


# Instantiate the main App
app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
