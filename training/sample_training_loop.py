#!/usr/bin/env python3
"""
Sample training loop for QRAgent_Bench with baseline factor approach.
"""

import sys
from pathlib import Path
from pprint import pprint

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from envs.factor_env import FactorImproveEnv


class SampleAgent:
    """Simple agent demonstrating factor improvement with baseline comparison."""
    
    def __init__(self):
        self.step_count = 0
        self.test_scenarios = [
            {"type": "OBSERVE", "tool": "describe_data"},
            {"type": "FACTOR_IMPROVE", "new_program": self._get_momentum_program()},
            {"type": "FACTOR_IMPROVE", "new_program": self._get_enhanced_momentum_program()},
            {"type": "FACTOR_IMPROVE", "new_program": self._get_mean_reversion_program()},
            {"type": "STOP"}
        ]
    
    def get_action(self, obs):
        """Get next action from predefined test scenarios."""
        if self.step_count >= len(self.test_scenarios):
            return {"type": "STOP"}
        
        action = self.test_scenarios[self.step_count]
        self.step_count += 1
        return action
    
    def _get_momentum_program(self):
        """Basic momentum factor."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 63},
                {"id": "score", "op": "zscore_xs", "src": "x0"}
            ],
            "output": "score"
        }
    
    def _get_enhanced_momentum_program(self):
        """Enhanced momentum with multiple timeframes."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "x1", "op": "rolling_return", "n": 21},
                {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                {"id": "x3", "op": "winsor_quantile", "src": "x2", "q": 0.02},
                {"id": "score", "op": "zscore_xs", "src": "x3"}
            ],
            "output": "score"
        }
    
    def _get_mean_reversion_program(self):
        """Mean reversion factor combining momentum and reversal."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "x1", "op": "rolling_return", "n": 21},
                {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                {"id": "x3", "op": "rolling_return", "n": 5},
                {"id": "x4", "op": "ema", "n": 10, "src": "x3"},
                {"id": "x5", "op": "sub", "a": "x3", "b": "x4"},
                {"id": "x6", "op": "add", "a": "x2", "b": "x5"},
                {"id": "x7", "op": "winsor_quantile", "src": "x6", "q": 0.02},
                {"id": "score", "op": "zscore_xs", "src": "x7"}
            ],
            "output": "score"
        }


def main():
    """Main training loop."""
    print("QRAgent_Bench - Sample Training Loop")
    
    # Initialize environment
    env = FactorImproveEnv(
        data_path="data/ff25_value_weighted.csv",
        test_train_split=0.8,
        timesteps=10,
        baseline_path="factors/baseline.json",
        plot_path="training_plots"
    )
    
    # Initialize agent
    agent = SampleAgent()
    obs, _ = env.reset()
    
    # Run episode
    total_reward = 0.0
    step = 0
    
    while True:
        step += 1
        action = agent.get_action(obs)
        
        obs, reward, terminated = env.step(action)
        total_reward += reward
        
        print(f"Step {step}: {action['type']} | Reward: {reward:.3f} | Total: {total_reward:.3f}")

        pprint(obs)
        
        if terminated:
            print(f"Episode Complete! Total Reward: {total_reward:.3f}")
            break


if __name__ == "__main__":
    main()