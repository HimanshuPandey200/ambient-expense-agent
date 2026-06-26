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

from app.agent import root_agent
from google.adk.workflow import Workflow


def test_workflow_graph_validation() -> None:
    """Verifies that the root_agent is a valid Workflow and compiles its graph."""
    assert isinstance(root_agent, Workflow)
    assert root_agent.graph is not None

    # This will raise an exception if the graph validation rules are violated.
    root_agent.graph.validate_graph()
