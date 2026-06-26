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

"""Integration tests for the expense agent workflow.

test_workflow_imports  — verifies the module loads, root_agent is a Workflow,
                         and the App is configured correctly. No LLM call.
test_workflow_graph    — re-verifies graph compiles (also tested in unit tests,
                         but good to catch import-level regressions here).
"""

from google.adk.workflow import Workflow
from google.adk.apps import App

from app.agent import root_agent, app as agent_app


def test_workflow_imports() -> None:
    """Verify the agent module loads without error and key objects are present."""
    assert root_agent is not None, "root_agent must not be None"
    assert isinstance(root_agent, Workflow), "root_agent must be a Workflow instance"
    assert isinstance(agent_app, App), "app must be an App instance"
    assert agent_app.name == "app", "App name must match the agent directory"


def test_workflow_graph() -> None:
    """Confirm the Workflow graph compiles and all four agent nodes are reachable."""
    assert root_agent.graph is not None
    root_agent.graph.validate_graph()

    node_names = {n.name for n in root_agent.graph.nodes}
    for expected in ("triage_agent", "security_agent", "policy_agent", "approval_agent"):
        assert expected in node_names, f"Node '{expected}' missing from graph"
