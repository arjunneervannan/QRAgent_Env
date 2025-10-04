#!/usr/bin/env python3
"""
Sample training loop for QRAgent_Bench with baseline factor approach.
Demonstrates the new improvement calculation and plotting functionality.
"""

import sys
from pathlib import Path
import json
import traceback

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
            print(f"🔍 DEBUG: No more test scenarios, returning STOP action")
            return {"type": "STOP"}
        
        action = self.test_scenarios[self.step_count]
        print(f"🔍 DEBUG: Getting action {self.step_count}: {action['type']}")
        
        # Validate action before returning
        is_valid, errors = validate_action(action)
        if not is_valid:
            print(f"❌ VALIDATION ERROR in action {self.step_count}:")
            for error in errors:
                print(f"   - {error}")
        else:
            print(f"✅ Action {self.step_count} validation passed")
        
        self.step_count += 1
        return action
    
    def _get_momentum_program(self):
        """Basic momentum factor (similar to baseline but different parameters)."""
        program = {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 63},  # Shorter lookback
                {"id": "score", "op": "zscore_xs", "src": "x0"}
            ],
            "output": "score"
        }
        
        # Debug: Validate program before returning
        print(f"🔍 DEBUG: Validating momentum program...")
        is_valid, errors = validate_program(program)
        if not is_valid:
            print(f"❌ MOMENTUM PROGRAM VALIDATION ERROR:")
            for error in errors:
                print(f"   - {error}")
        else:
            print(f"✅ Momentum program validation passed")
        
        return program
    
    def _get_enhanced_momentum_program(self):
        """Enhanced momentum with multiple timeframes."""
        program = {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "x1", "op": "rolling_return", "n": 21},
                {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                {"id": "x3", "op": "winsor_quantile", "src": "x2", "q": 0.02},
                {"id": "score", "op": "zscore_xs", "src": "x3"}
            ],
            "output": "score"
        }
        
        # Debug: Validate program before returning
        print(f"🔍 DEBUG: Validating enhanced momentum program...")
        is_valid, errors = validate_program(program)
        if not is_valid:
            print(f"❌ ENHANCED MOMENTUM PROGRAM VALIDATION ERROR:")
            for error in errors:
                print(f"   - {error}")
        else:
            print(f"✅ Enhanced momentum program validation passed")
        
        return program
    
    def _get_mean_reversion_program(self):
        """Mean reversion factor combining momentum and reversal."""
        program = {
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
        
        # Debug: Validate program before returning
        print(f"🔍 DEBUG: Validating mean reversion program...")
        is_valid, errors = validate_program(program)
        if not is_valid:
            print(f"❌ MEAN REVERSION PROGRAM VALIDATION ERROR:")
            for error in errors:
                print(f"   - {error}")
        else:
            print(f"✅ Mean reversion program validation passed")
        
        return program


def print_step_info(step, action, obs, reward, done):
    """Print concise step information with detailed debugging."""
    print(f"\n--- Step {step}: {action['type']} ---")
    print(f"🔍 DEBUG: Action details: {json.dumps(action, indent=2)}")
    print(f"Budget: {obs.get('budget_left', 'N/A')} | Reward: {reward:.3f} | Done: {done}")
    
    # Debug observation structure
    print(f"🔍 DEBUG: Observation keys: {list(obs.keys())}")
    
    if 'investment_performance' in obs:
        perf = obs['investment_performance']
        print(f"Strategy Sharpe: {perf.get('strategy_sharpe_net', 0):.3f} | "
              f"Baseline Sharpe: {perf.get('baseline_sharpe', 0):.3f} | "
              f"Improvement: {perf.get('improvement', 0):.3f}")
        if 'plot_path' in perf:
            print(f"Plot saved: {perf['plot_path']}")
        print(f"🔍 DEBUG: Investment performance keys: {list(perf.keys())}")
    
    if 'observation_result' in obs:
        result = obs['observation_result']
        print(f"🔍 DEBUG: Observation result type: {type(result)}")
        if isinstance(result, dict) and 'shape' in result:
            print(f"Data shape: {result['shape']}")
        elif isinstance(result, dict):
            print(f"🔍 DEBUG: Observation result keys: {list(result.keys())}")
        else:
            print(f"🔍 DEBUG: Observation result: {result}")
    
    # Debug any error information
    if 'error' in obs:
        print(f"❌ ERROR in observation: {obs['error']}")
    
    if 'info' in obs:
        print(f"🔍 DEBUG: Info: {obs['info']}")


def main():
    """Main training loop demonstrating baseline factor approach."""
    print("🚀 QRAgent_Bench - Sample Training Loop")
    print("="*50)
    
    # Check baseline file first
    baseline_path = "/factors/baseline.json"
    print(f"🔍 DEBUG: Checking baseline file: {baseline_path}")
    try:
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        print(f"✅ Baseline file loaded successfully")
        print(f"🔍 DEBUG: Baseline data keys: {list(baseline_data.keys())}")
        
        # Validate baseline program
        if 'program' in baseline_data:
            print(f"🔍 DEBUG: Validating baseline program...")
            is_valid, errors = validate_program(baseline_data['program'])
            if not is_valid:
                print(f"❌ BASELINE PROGRAM VALIDATION ERROR:")
                for error in errors:
                    print(f"   - {error}")
            else:
                print(f"✅ Baseline program validation passed")
        else:
            print(f"⚠️ WARNING: No 'program' key in baseline data")
            
    except FileNotFoundError:
        print(f"❌ Baseline file not found: {baseline_path}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Baseline file JSON decode error: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading baseline file: {e}")
        return
    
    # Initialize environment with baseline factor
    try:
        print("🔍 DEBUG: Initializing environment...")
        print(f"🔍 DEBUG: Data path: data/ff25_daily.csv")
        print(f"🔍 DEBUG: Baseline path: {baseline_path}")
        print(f"🔍 DEBUG: Plot path: training_plots")
        
        env = FactorImproveEnv(
            data_path="data/ff25_value_weighted.csv",
            test_train_split=0.8,
            timesteps=10,
            baseline_path=baseline_path,
            plot_path="training_plots"
        )
        print(f"✅ Environment initialized (Data: {env.returns.shape})")
        print(f"🔍 DEBUG: Environment attributes: {[attr for attr in dir(env) if not attr.startswith('_')]}")
    except Exception as e:
        print(f"❌ Environment failed: {e}")
        print(f"🔍 DEBUG: Full traceback:")
        traceback.print_exc()
        return
    
    # Initialize sample agent
    print("🔍 DEBUG: Initializing sample agent...")
    agent = SampleAgent()
    print(f"🔍 DEBUG: Agent test scenarios: {len(agent.test_scenarios)}")
    
    try:
        print("🔍 DEBUG: Resetting environment...")
        obs, info = env.reset()
        print(f"✅ Episode started (Budget: {obs['budget_left']})")
        print(f"🔍 DEBUG: Reset info: {info}")
        print(f"🔍 DEBUG: Initial observation keys: {list(obs.keys())}")
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        print(f"🔍 DEBUG: Reset traceback:")
        traceback.print_exc()
        return
    
    # Run episode
    total_reward = 0.0
    step = 0
    
    while True:
        step += 1
        print(f"\n🔍 DEBUG: Starting step {step}...")
        
        try:
            action = agent.get_action(obs)
            print(f"🔍 DEBUG: Executing action: {action['type']}")
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"🔍 DEBUG: Step completed - Reward: {reward}, Done: {done}, Truncated: {truncated}")
            print(f"🔍 DEBUG: Step info: {info}")
            
            print_step_info(step, action, obs, reward, done)
            
            if done or truncated:
                print(f"\n🏁 Episode Complete!")
                print(f"Total Reward: {total_reward:.3f} | Steps: {step}")
                print(f"🔍 DEBUG: Final observation keys: {list(obs.keys())}")
                break
                
        except Exception as e:
            print(f"❌ Step {step} failed: {e}")
            print(f"🔍 DEBUG: Step {step} traceback:")
            traceback.print_exc()
            print(f"🔍 DEBUG: Current observation: {obs}")
            print(f"🔍 DEBUG: Current action: {action}")
            break
    
    print(f"\n{'='*50}")
    print("Training Complete!")


if __name__ == "__main__":
    main()
