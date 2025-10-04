#!/usr/bin/env python3
"""
Sample training loop for QRAgent_Bench with baseline factor approach.
Demonstrates the new improvement calculation and plotting functionality.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from envs.factor_env import FactorImproveEnv
from factors.validate import validate_program, validate_action


class SampleAgent:
    """Simple agent demonstrating factor improvement with baseline comparison."""
    
    def __init__(self):
        self.step_count = 0
        # Progressive factor improvements to demonstrate the system
        self.test_scenarios = [
            # Step 1: Observe data
            {"type": "OBSERVE", "tool": "describe_data"},
            
            # Step 2: First factor improvement (vs baseline)
            {"type": "FACTOR_IMPROVE", "new_program": self._get_momentum_program()},
            
            # Step 3: Second factor improvement (vs previous)
            {"type": "FACTOR_IMPROVE", "new_program": self._get_enhanced_momentum_program()},
            
            # Step 4: Third factor improvement (vs previous)
            {"type": "FACTOR_IMPROVE", "new_program": self._get_mean_reversion_program()},
            
            # Step 5: Final evaluation
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
        """Basic momentum factor (similar to baseline but different parameters)."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 63},  # Shorter lookback
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


def print_step_info(step, action, obs, reward, done):
    """Print concise step information."""
    print(f"\n--- Step {step}: {action['type']} ---")
    print(f"Budget: {obs.get('budget_left', 'N/A')} | Reward: {reward:.3f} | Done: {done}")
    
    if 'investment_performance' in obs:
        perf = obs['investment_performance']
        print(f"Strategy Sharpe: {perf.get('strategy_sharpe_net', 0):.3f} | "
              f"Baseline Sharpe: {perf.get('baseline_sharpe', 0):.3f} | "
              f"Improvement: {perf.get('improvement', 0):.3f}")
        if 'plot_path' in perf:
            print(f"Plot saved: {perf['plot_path']}")
    
    if 'observation_result' in obs:
        result = obs['observation_result']
        if isinstance(result, dict) and 'shape' in result:
            print(f"Data shape: {result['shape']}")


def main():
    """Main training loop demonstrating baseline factor approach."""
    print("🚀 QRAgent_Bench - Sample Training Loop")
    print("="*50)
    
    # Initialize environment with baseline factor
    try:
        env = FactorImproveEnv(
            data_path="data/ff25_daily.csv",
            test_train_split=0.8,
            timesteps=10,
            baseline_path="baseline.json",
            plot_path="training_plots"
        )
        print(f"✅ Environment initialized (Data: {env.returns.shape})")
    except Exception as e:
        print(f"❌ Environment failed: {e}")
        return
    
    # Initialize sample agent
    agent = SampleAgent()
    obs, _ = env.reset()
    print(f"✅ Episode started (Budget: {obs['budget_left']})")
    
    # Run episode
    total_reward = 0.0
    step = 0
    
    while True:
        step += 1
        action = agent.get_action(obs)
        
        try:
            obs, reward, done = env.step(action)
            total_reward += reward
            print_step_info(step, action, obs, reward, done)
            
            if done:
                print(f"\n🏁 Episode Complete!")
                print(f"Total Reward: {total_reward:.3f} | Steps: {step}")
                break
                
        except Exception as e:
            print(f"❌ Step failed: {e}")
            break
    
    print(f"\n{'='*50}")
    print("Training Complete!")


if __name__ == "__main__":
    main()
