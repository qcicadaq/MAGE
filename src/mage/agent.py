import os
import re
import sys
import traceback
from typing import List, Tuple

from llama_index.core.llms import LLM

from .log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout
from .mcp_agent import MCPAgent, ProblemState, Action
from .rtl_editor import RTLEditor
from .rtl_generator import RTLGenerator
from .sim_judge import SimJudge
from .sim_reviewer import SimReviewer
from .tb_generator import TBGenerator
from .token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)


class TopAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.token_counter = (
            TokenCounterCached(llm)
            if TokenCounterCached.is_cache_enabled(llm)
            else TokenCounter(llm)
        )
        self.max_mcp_iterations = 10  # Max iterations for the MCP loop
        self.rtl_max_candidates = 20
        self.rtl_selected_candidates = 2
        self.redirect_log = False
        self.output_path = "./output"
        self.log_path = "./log"
        self.golden_tb_path: str | None = None
        self.golden_rtl_blackbox_path: str | None = None

        # Sub-agents will be initialized in execute_action
        self.tb_gen: TBGenerator | None = None
        self.rtl_gen: RTLGenerator | None = None
        self.sim_reviewer: SimReviewer | None = None
        self.sim_judge: SimJudge | None = None
        self.rtl_edit: RTLEditor | None = None

        # Initialize the MCP Agent
        self.mcp_agent = MCPAgent(self.llm)

    def set_output_path(self, output_path: str) -> None:
        self.output_path = output_path

    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    def set_redirect_log(self, new_value: bool) -> None:
        self.redirect_log = new_value
        if self.redirect_log:
            switch_log_to_file()
        else:
            switch_log_to_stdout()

    def write_output(self, content: str, file_name: str) -> None:
        assert self.output_dir_per_run
        with open(f"{self.output_dir_per_run}/{file_name}", "w") as f:
            f.write(content)

    def initialize_state(self, spec: str) -> ProblemState:
        return ProblemState(spec=spec)

    def execute_action(self, action: Action, state: ProblemState) -> ProblemState:
        logger.info(f"Executing action: {action.name}")
        state.history.append(action)
        state.iteration += 1

        if action == Action.GENERATE_TESTBENCH:
            if not self.tb_gen:
                self.tb_gen = TBGenerator(self.token_counter)
            self.tb_gen.reset()
            self.tb_gen.set_golden_tb_path(self.golden_tb_path)
            if not self.golden_tb_path:
                logger.info("No golden testbench provided")
            
            # Logic to handle fixing the testbench
            if state.simulation_log:
                 self.tb_gen.set_failed_trial(state.simulation_log, state.rtl_code, state.testbench)

            testbench, interface = self.tb_gen.chat(state.spec)
            state.testbench = testbench
            state.interface = interface
            logger.info("Generated tb:\n" + testbench)
            logger.info("Generated if:\n" + interface)
            self.write_output(testbench, "tb.sv")
            self.write_output(interface, "if.sv")

        elif action == Action.GENERATE_INITIAL_RTL:
            if not self.rtl_gen:
                self.rtl_gen = RTLGenerator(self.token_counter)
            self.rtl_gen.reset()
            is_syntax_pass, rtl_code = self.rtl_gen.chat(
                input_spec=state.spec,
                testbench=state.testbench,
                interface=state.interface,
                rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv"),
            )
            if not is_syntax_pass:
                state.rtl_code = rtl_code
                state.is_simulation_pass = False # Syntax failure is a failure
            else:
                state.rtl_code = rtl_code
                self.write_output(rtl_code, "rtl.sv")
                logger.info("Initial rtl:\n" + rtl_code)

        elif action == Action.RUN_SIMULATION:
            if not self.sim_reviewer:
                self.sim_reviewer = SimReviewer(self.output_dir_per_run, self.golden_rtl_blackbox_path)
            is_pass, mismatch_count, log = self.sim_reviewer.review()
            state.is_simulation_pass = is_pass
            state.simulation_mismatch_count = mismatch_count
            state.simulation_log = log

        elif action == Action.ANALYZE_SIMULATION_FAILURE:
            if not self.sim_judge:
                self.sim_judge = SimJudge(self.token_counter)
            self.sim_judge.reset()
            tb_needs_fix = self.sim_judge.chat(state.spec, state.simulation_log, state.rtl_code, state.testbench)
            state.tb_needs_fix = tb_needs_fix
            state.rtl_needs_fix = not tb_needs_fix

        elif action == Action.GENERATE_RTL_CANDIDATES:
            if not self.rtl_gen:
                self.rtl_gen = RTLGenerator(self.token_counter)
            self.rtl_gen.reset()
            candidates = self.rtl_gen.gen_candidates(
                input_spec=state.spec,
                testbench=state.testbench,
                interface=state.interface,
                rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv"),
                candidates_num=self.rtl_max_candidates,
                enable_cache=True,
            )
            
            candidates_info = []
            for is_syntax_pass, rtl_code_candidate in candidates:
                if not is_syntax_pass:
                    continue
                self.write_output(rtl_code_candidate, "rtl.sv")
                is_sim_pass, sim_mismatch, sim_log = self.sim_reviewer.review()
                if is_sim_pass:
                    state.is_simulation_pass = True
                    state.rtl_code = rtl_code_candidate
                    return state # Early exit on success
                candidates_info.append((rtl_code_candidate, sim_mismatch, sim_log))

            candidates_info.sort(key=lambda x: x[1])
            state.candidates_info = candidates_info
            state.is_simulation_pass = False

        elif action == Action.EDIT_RTL_WITH_FEEDBACK:
            if not self.rtl_edit:
                self.rtl_edit = RTLEditor(self.token_counter, sim_reviewer=self.sim_reviewer)
            
            # Simplified: just try the top candidate
            if state.candidates_info:
                top_candidate_code, top_candidate_mismatch, top_candidate_log = state.candidates_info[0]
                self.write_output(top_candidate_code, "rtl.sv")
                
                self.rtl_edit.reset()
                is_pass, rtl_code = self.rtl_edit.chat(
                    spec=state.spec,
                    output_dir_per_run=self.output_dir_per_run,
                    sim_failed_log=top_candidate_log,
                    sim_mismatch_cnt=top_candidate_mismatch,
                )
                state.is_simulation_pass = is_pass
                state.rtl_code = rtl_code
            else:
                logger.warning("EDIT_RTL_WITH_FEEDBACK called with no candidates.")
                state.is_simulation_pass = False


        return state

    def run_instance(self, spec: str) -> Tuple[bool, str]:
        """
        Run a single instance of the benchmark using the MCP agent.
        Return value:
        - is_pass: bool, whether the instance passes the golden testbench
        - rtl_code: str, the generated RTL code
        """
        mcp_agent = MCPAgent(self.llm)
        state = self.initialize_state(spec)

        for _ in range(self.max_mcp_iterations):
            action = mcp_agent.get_next_action(state)

            if action in [Action.FINISH_SUCCESS, Action.FINISH_FAILURE]:
                logger.info(f"MCP decided to finish with status: {action.name}")
                break
            
            state = self.execute_action(action, state)

        # Final check
        if not self.sim_reviewer:
             self.sim_reviewer = SimReviewer(self.output_dir_per_run, self.golden_rtl_blackbox_path)
        is_pass, _, _ = self.sim_reviewer.review()

        return is_pass, state.rtl_code or ""

    def _run(self, spec: str) -> Tuple[bool, str]:
        try:
            if os.path.exists(f"{self.output_dir_per_run}/properly_finished.tag"):
                os.remove(f"{self.output_dir_per_run}/properly_finished.tag")
            self.token_counter.reset()
            # Initialize sim_reviewer here as it's used in run_instance and execute_action
            self.sim_reviewer = SimReviewer(
                self.output_dir_per_run,
                self.golden_rtl_blackbox_path,
            )
            
            ret = self.run_instance(spec)

            self.token_counter.log_token_stats()
            with open(f"{self.output_dir_per_run}/properly_finished.tag", "w") as f:
                f.write("1")
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            ret = False, f"Exception: {exc_info[1]}"
        return ret

    def run(
        self,
        benchmark_type_name: str,
        task_id: str,
        spec: str,
        golden_tb_path: str | None = None,
        golden_rtl_blackbox_path: str | None = None,
    ) -> Tuple[bool, str]:
        self.golden_tb_path = golden_tb_path
        self.golden_rtl_blackbox_path = golden_rtl_blackbox_path
        log_dir_per_run = f"{self.log_path}/{benchmark_type_name}_{task_id}"
        self.output_dir_per_run = f"{self.output_path}/{benchmark_type_name}_{task_id}"
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_dir_per_run, exist_ok=True)
        set_log_dir(log_dir_per_run)
        if self.redirect_log:
            with open(f"{log_dir_per_run}/mage_rtl.log", "w") as f:
                sys.stdout = f
                sys.stderr = f
                result = self._run(spec)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        else:
            result = self._run(spec)
        # Redirect log contains format with rich text.
        # Provide a rich-free version for log parsing or less viewing.
        if self.redirect_log:
            with open(f"{log_dir_per_run}/mage_rtl.log", "r") as f:
                content = f.read()
            content = re.sub(r"\[.*?m", "", content)
            with open(f"{log_dir_per_run}/mage_rtl_rich_free.log", "w") as f:
                f.write(content)
        return result
