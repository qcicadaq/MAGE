import argparse
import os
import sys
from datetime import timedelta
import time

from mage.agent import TopAgent
from mage.gen_config import get_llm, set_exp_setting
from mage.log_utils import get_logger
from mage.mcp_agent import Action

logger = get_logger(__name__)

# Configuration for the chat session
args_dict = {
    "provider": "openrouter",
    "model": "openai/gpt-oss-120b:free",
    "temperature": 0.85,
    "top_p": 0.95,
    "max_token": 8192,
    "key_cfg_path": "./key.cfg",
    "run_identifier": "interactive_chat",
}


def print_state_summary(state):
    """Prints a summary of the current problem state."""
    print("\n" + "="*50)
    print("CURRENT STATE:")
    print(f"  - Last action: {state.history[-1] if state.history else 'None'}")
    print(f"  - Simulation status: {'PASS' if state.is_simulation_pass else ('FAIL' if state.is_simulation_pass is False else 'Not run')}")
    if state.simulation_mismatch_count is not None:
        print(f"  - Mismatches: {state.simulation_mismatch_count}")
    print("="*50 + "\n")

def run_non_interactive(agent: TopAgent, mcp_agent, spec: str):
    """Runs the MCP loop non-interactively."""
    state = agent.initialize_state(spec)
    logger.info(f"Running non-interactively with prompt: {spec}")

    while state.iteration < agent.max_mcp_iterations:
        action = mcp_agent.get_next_action(state)
        logger.info(f"MCP Action: {action.name}")

        if action in [Action.FINISH_SUCCESS, Action.FINISH_FAILURE]:
            logger.info(f"MCP finished with status: {action.name}")
            break
        
        state = agent.execute_action(action, state)

    is_pass, rtl_code = state.is_simulation_pass, state.rtl_code
    print("\n" + "="*50)
    print("NON-INTERACTIVE RUN COMPLETE")
    print(f"Final Status: {'PASS' if is_pass else 'FAIL'}")
    print("Generated RTL:")
    print(rtl_code or "No RTL was generated.")
    print("="*50 + "\n")


def run_interactive(agent: TopAgent, mcp_agent):
    """Runs the MCP loop interactively."""
    print("Welcome to the interactive MAGE chat.")
    print("Please provide the specification for the RTL you want to generate.")
    spec = input(">> ")
    state = agent.initialize_state(spec)

    while state.iteration < agent.max_mcp_iterations:
        action = mcp_agent.get_next_action(state)
        print(f"🤖 The MCP agent decided to take the following action: {action.name}")

        if action in [Action.FINISH_SUCCESS, Action.FINISH_FAILURE]:
            print(f"🏁 The MCP has finished with status: {action.name}")
            break

        user_input = input("   Press Enter to continue, or type 'q' to quit. ")
        if user_input.lower() == 'q':
            print("Exiting chat.")
            break

        start_time = time.monotonic()
        state = agent.execute_action(action, state)
        run_time = timedelta(seconds=time.monotonic() - start_time)

        print_state_summary(state)
        print(f"Action '{action.name}' took {run_time} to execute.")


def main():
    """Main function to run the interactive chat loop."""
    parser = argparse.ArgumentParser(description="MAGE Interactive Chat")
    parser.add_argument("--prompt", type=str, help="Run non-interactively with a specific prompt.")
    cli_args = parser.parse_args()

    # Merge config
    for key, value in args_dict.items():
        if not hasattr(cli_args, key) or getattr(cli_args, key) is None:
            setattr(cli_args, key, value)
    
    args = cli_args

    llm = get_llm(
        model=args.model,
        cfg_path=args.key_cfg_path,
        max_token=args.max_token,
        provider=args.provider,
    )
    set_exp_setting(temperature=args.temperature, top_p=args.top_p)

    agent = TopAgent(llm)
    agent.set_output_path(f"./output_{args.run_identifier}")
    agent.set_log_path(f"./log_{args.run_identifier}")
    agent.set_redirect_log(False) 

    # Manually set up the directories for the run
    task_id = "interactive_run"
    benchmark_type_name = "chat"
    agent.output_dir_per_run = f"{agent.output_path}/{benchmark_type_name}_{task_id}"
    os.makedirs(agent.output_dir_per_run, exist_ok=True)
    agent.token_counter.reset()

    mcp_agent = agent.mcp_agent
    
    if args.prompt:
        run_non_interactive(agent, mcp_agent, args.prompt)
    else:
        try:
            run_interactive(agent, mcp_agent)
        except EOFError:
            logger.error("Cannot run in interactive mode in a non-interactive environment.")
            logger.error("Please use the --prompt argument to run non-interactively.")
            logger.error('Example: .venv/bin/python chat.py --prompt "Create a 2-input AND gate"')


    print("\nFinal summary:")
    agent.token_counter.log_token_stats()


if __name__ == "__main__":
    main()
