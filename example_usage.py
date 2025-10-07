#!/usr/bin/env python3
"""
Example usage of the QRAgent_Bench system.
Demonstrates factor improvement and backtesting capabilities.
"""

import json
from envs.factor_env import FactorImproveEnv
from engine.data_analysis import describe_data, plot_returns, analyze_factor_performance

def demonstrate_basic_usage():
    """Demonstrate basic environment usage."""
    print("=== Basic Environment Usage ===\n")
    
    # Initialize environment
    env = FactorImproveEnv(
        data_path="data/ff25_value_weighted.csv",
        test_train_split=0.8,
        timesteps=10,
        baseline_path="factors/baseline.json",
        plot_path="example_plots"
    )
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial state: Budget={obs['budget_left']}")
    print(f"Current program nodes: {len(obs['current_program']['nodes'])}")
    
    return env, obs

def demonstrate_observation_tools():
    """Demonstrate observation tools."""
    print("\n=== Observation Tools ===\n")
    
    env, obs = demonstrate_basic_usage()
    
    # Step 1: Observe data
    print("--- Step 1: Describing Data ---")
    action = {"type": "OBSERVE", "tool": "describe_data"}
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action['type']} - {action['tool']}")
    print(f"Reward: {reward:.3f}, Budget: {obs['budget_left']}")
    
    if 'observation_result' in obs:
        result = obs['observation_result']
        print(f"Data shape: {result.get('shape', 'N/A')}")
        print(f"Date range: {result.get('date_range', 'N/A')}")
    
    return env, obs

def demonstrate_factor_improvement():
    """Demonstrate factor improvement."""
    print("\n=== Factor Improvement ===\n")
    
    env, obs = demonstrate_observation_tools()
    
    # Step 2: Improve factor
    print("--- Step 2: Improving Factor ---")
    new_program = {
        "nodes": [
            # Momentum signal
            {"id": "x0", "op": "rolling_return", "n": 126},
            {"id": "x1", "op": "rolling_return", "n": 21},
            {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
            
            # Mean reversion signal
            {"id": "x3", "op": "rolling_return", "n": 5},
            {"id": "x4", "op": "ema", "n": 10, "src": "x3"},
            {"id": "x5", "op": "sub", "a": "x3", "b": "x4"},
            
            # Combine signals
            {"id": "x6", "op": "add", "a": "x2", "b": "x5"},
            
            # Final processing
            {"id": "x7", "op": "winsor_quantile", "src": "x6", "q": 0.02},
            {"id": "score", "op": "zscore_xs", "src": "x7"}
        ],
        "output": "score"
    }
    
    action = {"type": "FACTOR_IMPROVE", "new_program": new_program}
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action['type']}")
    print(f"Reward: {reward:.3f}, Budget: {obs['budget_left']}")
    
    if 'investment_performance' in obs:
        perf = obs['investment_performance']
        print(f"Strategy Sharpe: {perf.get('strategy_sharpe_net', 0):.3f}")
        print(f"Baseline Sharpe: {perf.get('baseline_sharpe', 0):.3f}")
        print(f"Improvement: {perf.get('improvement', 0):.3f}")
    
    return env, obs

def demonstrate_evaluation():
    """Demonstrate final evaluation."""
    print("\n=== Final Evaluation ===\n")
    
    env, obs = demonstrate_factor_improvement()
    
    # Step 3: Stop (triggers automatic evaluation)
    print("--- Step 3: Stopping (triggers evaluation) ---")
    action = {"type": "STOP"}
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action['type']}")
    print(f"Reward: {reward:.3f}, Terminated: {terminated}")
    
    if 'investment_performance' in obs:
        perf = obs['investment_performance']
        print(f"Final OOS Sharpe: {perf.get('strategy_sharpe_net', 0):.3f}")
        print(f"Turnover: {perf.get('turnover', 0):.3f}")
        if 'plot_path' in perf:
            print(f"Plot saved: {perf['plot_path']}")
    
    print(f"\nEpisode Summary:")
    print(f"  Total reward: {sum(obs.get('episode_rewards', [])):.3f}")
    print(f"  Steps taken: {len(obs.get('episode_rewards', []))}")

def demonstrate_custom_reward_config():
    """Demonstrate custom reward configuration."""
    print("\n=== Custom Reward Configuration ===\n")
    
    # Create custom reward config
    custom_config = {
        "factor_improve": {
            "base_reward_multiplier": 3.0,
            "improvement_formula": "current_sharpe - equal_weight_sharpe"
        },
        "observe": {
            "success_reward": 0.2,
            "failure_reward": -5.0
        },
        "stop": {
            "base_multiplier": 1.0,
            "cost_per_turnover": 0.05,
            "cost_per_step": 0.02,
            "pass_guard": 1.0,
            "fail_guard": -2.0
        },
        "validation_error": {
            "reward": -5.0
        }
    }
    
    # Save custom config
    with open("custom_reward_config.json", "w") as f:
        json.dump(custom_config, f, indent=2)
    
    print("Custom reward configuration created: custom_reward_config.json")
    print("Key changes:")
    print("  - Higher factor improvement multiplier (3.0 vs 2.0)")
    print("  - Higher observation success reward (0.2 vs 0.1)")
    print("  - Lower validation error penalty (-5.0 vs -10.0)")

def demonstrate_factor_validation():
    """Demonstrate factor program validation."""
    print("\n=== Factor Validation ===\n")
    
    from factors.validate import validate_program
    
    # Valid program
    valid_program = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 63},
            {"id": "score", "op": "zscore_xs", "src": "x0"}
        ],
        "output": "score"
    }
    
    is_valid, errors = validate_program(valid_program)
    print(f"Valid program: {is_valid}")
    if not is_valid:
        for error in errors:
            print(f"  Error: {error}")
    
    # Invalid program (missing required parameter)
    invalid_program = {
        "nodes": [
            {"id": "x0", "op": "rolling_return"},  # Missing 'n' parameter
            {"id": "score", "op": "zscore_xs", "src": "x0"}
        ],
        "output": "score"
    }
    
    is_valid, errors = validate_program(invalid_program)
    print(f"Invalid program: {is_valid}")
    if not is_valid:
        for error in errors:
            print(f"  Error: {error}")

if __name__ == "__main__":
    print("QRAgent_Bench - Factor Strategy Development Environment")
    print("=" * 60)
    
    demonstrate_basic_usage()
    demonstrate_observation_tools()
    demonstrate_factor_improvement()
    demonstrate_evaluation()
    demonstrate_custom_reward_config()
    demonstrate_factor_validation()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")