from enum import Enum
from typing import List, Optional, Tuple

from llama_index.core.llms import LLM
from pydantic import BaseModel, Field

from .log_utils import get_logger

logger = get_logger(__name__)


class Action(str, Enum):
    """Enumeration of actions the MCP agent can take."""

    GENERATE_TESTBENCH = "generate_testbench"
    GENERATE_INITIAL_RTL = "generate_initial_rtl"
    RUN_SIMULATION = "run_simulation"
    ANALYZE_SIMULATION_FAILURE = "analyze_simulation_failure"
    GENERATE_RTL_CANDIDATES = "generate_rtl_candidates"
    EDIT_RTL_WITH_FEEDBACK = "edit_rtl_with_feedback"
    FINISH_SUCCESS = "finish_success"
    FINISH_FAILURE = "finish_failure"


class ProblemState(BaseModel):
    """Data model for the state of the problem-solving process."""

    spec: str
    history: List[Action] = Field(default_factory=list)
    iteration: int = 0

    # Agent-specific data
    testbench: Optional[str] = None
    interface: Optional[str] = None
    rtl_code: Optional[str] = None
    simulation_log: Optional[str] = None
    simulation_mismatch_count: Optional[int] = None
    is_simulation_pass: Optional[bool] = None
    analysis_of_failure: Optional[str] = None
    fix_suggestion: Optional[str] = None
    tb_needs_fix: bool = False
    rtl_needs_fix: bool = False

    candidates_info: List[Tuple[str, int, str]] = Field(default_factory=list)

    def is_successful(self) -> bool:
        """Returns True if the problem is solved."""
        return self.is_simulation_pass is True


class MCPAgent:
    """The Master Control Program agent."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def get_next_action(self, state: ProblemState) -> Action:
        """
        Determines the next action to take based on the current state.
        This is currently a rule-based implementation.
        """
        if not state.history:
            return Action.GENERATE_TESTBENCH

        last_action = state.history[-1]

        if last_action == Action.GENERATE_TESTBENCH:
            return Action.GENERATE_INITIAL_RTL

        if last_action == Action.GENERATE_INITIAL_RTL:
            return Action.RUN_SIMULATION

        if last_action == Action.RUN_SIMULATION:
            if state.is_simulation_pass:
                return Action.FINISH_SUCCESS
            else:
                return Action.ANALYZE_SIMULATION_FAILURE
        
        if last_action == Action.ANALYZE_SIMULATION_FAILURE:
            if state.tb_needs_fix:
                # In the original logic, the testbench is fixed and then the loop continues.
                # Here we will try to generate a new testbench.
                return Action.GENERATE_TESTBENCH
            else:
                return Action.GENERATE_RTL_CANDIDATES

        if last_action == Action.GENERATE_RTL_CANDIDATES:
            if state.is_simulation_pass: # A candidate might have passed
                return Action.FINISH_SUCCESS
            elif state.candidates_info:
                return Action.EDIT_RTL_WITH_FEEDBACK
            else:
                # No passing candidates and no info to edit from
                return Action.FINISH_FAILURE
        
        if last_action == Action.EDIT_RTL_WITH_FEEDBACK:
            if state.is_simulation_pass:
                return Action.FINISH_SUCCESS
            else:
                # For simplicity, we'll try to generate new candidates if editing fails.
                # A more complex strategy could be implemented here.
                if state.iteration < 3: # Limit retries
                    return Action.GENERATE_RTL_CANDIDATES
                else:
                    return Action.FINISH_FAILURE

        # Default fallback
        logger.warning("Could not determine next action, finishing with failure.")
        return Action.FINISH_FAILURE

