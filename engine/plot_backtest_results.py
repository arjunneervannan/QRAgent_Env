from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_strategy_results(strategy_weights: pd.DataFrame, 
                         strategy_net_returns: pd.Series, 
                         strategy_gross_returns: pd.Series,
                         baseline_weights: pd.DataFrame,
                         baseline_net_returns: pd.Series,
                         returns: pd.DataFrame,
                         title: str = "Strategy Results", 
                         plot_path: str = None) -> str:
    """Generate comprehensive 2x2 grid plots for strategy evaluation."""
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle(title, fontsize=16)
    
    # Calculate baseline returns
    baseline_returns = (baseline_weights * returns).sum(axis=1)
    
    # Calculate basic metrics manually (no metrics import)
    strategy_sharpe_net = strategy_net_returns.mean() / strategy_net_returns.std() * np.sqrt(252) if strategy_net_returns.std() > 0 else 0
    strategy_sharpe_gross = strategy_gross_returns.mean() / strategy_gross_returns.std() * np.sqrt(252) if strategy_gross_returns.std() > 0 else 0
    baseline_sharpe_net = baseline_net_returns.mean() / baseline_net_returns.std() * np.sqrt(252) if baseline_net_returns.std() > 0 else 0
    
    # Calculate improvement
    improvement = strategy_sharpe_net - baseline_sharpe_net
    
    # Calculate additional metrics
    strategy_ann_return = strategy_net_returns.mean() * 252
    strategy_ann_vol = strategy_net_returns.std() * np.sqrt(252)
    baseline_ann_return = baseline_net_returns.mean() * 252
    baseline_ann_vol = baseline_net_returns.std() * np.sqrt(252)
    
    # Calculate max drawdown
    strategy_cumulative = (1 + strategy_net_returns).cumprod()
    strategy_peaks = strategy_cumulative.cummax()
    strategy_drawdown = (strategy_cumulative / strategy_peaks) - 1
    strategy_max_dd = strategy_drawdown.min()
    
    baseline_cumulative = (1 + baseline_net_returns).cumprod()
    baseline_peaks = baseline_cumulative.cummax()
    baseline_drawdown = (baseline_cumulative / baseline_peaks) - 1
    baseline_max_dd = baseline_drawdown.min()
    
    # 1. Net performance, gross performance, and baseline factor
    ax1 = axes[0, 0]
    cumulative_net = (1 + strategy_net_returns).cumprod()
    cumulative_gross = (1 + strategy_gross_returns).cumprod()
    cumulative_baseline = (1 + baseline_net_returns).cumprod()
    
    cumulative_net.plot(ax=ax1, label="Strategy Net", alpha=0.8, linewidth=2, color='blue')
    cumulative_gross.plot(ax=ax1, label="Strategy Gross", alpha=0.8, linewidth=2, color='lightblue')
    cumulative_baseline.plot(ax=ax1, label="Baseline Factor", alpha=0.8, linestyle='--', linewidth=2, color='orange')
    
    ax1.set_title("Cumulative Performance")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 252-day rolling improvement (strategy vs baseline)
    ax2 = axes[0, 1]
    rolling_window = 252
    if len(strategy_net_returns) >= rolling_window:
        rolling_improvement = []
        for i in range(rolling_window, len(strategy_net_returns)):
            strategy_window = strategy_net_returns.iloc[i-rolling_window:i]
            baseline_window = baseline_net_returns.iloc[i-rolling_window:i]
            strategy_sharpe = strategy_window.mean() / strategy_window.std() * np.sqrt(252) if strategy_window.std() > 0 else 0
            baseline_sharpe = baseline_window.mean() / baseline_window.std() * np.sqrt(252) if baseline_window.std() > 0 else 0
            improvement = strategy_sharpe - baseline_sharpe
            rolling_improvement.append(improvement)
        
        rolling_improvement_series = pd.Series(rolling_improvement, index=strategy_net_returns.index[rolling_window:])
        rolling_improvement_series.plot(ax=ax2, color='green', alpha=0.8, linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    ax2.set_title("252-Day Rolling Improvement (vs Baseline)")
    ax2.set_ylabel("Sharpe Improvement")
    ax2.grid(True, alpha=0.3)
    
    # 3. 252-day rolling Sharpe for strategy and baseline factor
    ax3 = axes[1, 0]
    if len(strategy_net_returns) >= rolling_window:
        strategy_rolling_sharpe = strategy_net_returns.rolling(rolling_window).mean() / strategy_net_returns.rolling(rolling_window).std() * np.sqrt(252)
        baseline_rolling_sharpe = baseline_net_returns.rolling(rolling_window).mean() / baseline_net_returns.rolling(rolling_window).std() * np.sqrt(252)
        
        strategy_rolling_sharpe.plot(ax=ax3, color='blue', alpha=0.8, linewidth=2, label="Strategy")
        baseline_rolling_sharpe.plot(ax=ax3, color='orange', alpha=0.8, linewidth=2, linestyle='--', label="Baseline Factor")
    
    ax3.set_title("252-Day Rolling Sharpe Ratio")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Drawdown graph
    ax4 = axes[1, 1]
    # Strategy drawdown
    strategy_drawdown.plot(ax=ax4, color='red', alpha=0.8, linewidth=2, label="Strategy")
    ax4.fill_between(strategy_drawdown.index, strategy_drawdown, 0, alpha=0.3, color='red')
    
    # Baseline drawdown
    baseline_drawdown.plot(ax=ax4, color='orange', alpha=0.8, linewidth=2, linestyle='--', label="Baseline Factor")
    ax4.fill_between(baseline_drawdown.index, baseline_drawdown, 0, alpha=0.2, color='orange')
    
    ax4.set_title("Drawdown")
    ax4.set_ylabel("Drawdown")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add comprehensive metrics summary as text overlay
    metrics_text = (
        f"STRATEGY PERFORMANCE:\n"
        f"  Net Sharpe: {strategy_sharpe_net:.3f}\n"
        f"  Gross Sharpe: {strategy_sharpe_gross:.3f}\n"
        f"  Ann. Return: {strategy_ann_return:.1%}\n"
        f"  Ann. Vol: {strategy_ann_vol:.1%}\n"
        f"  Max DD: {strategy_max_dd:.1%}\n\n"
        f"BASELINE PERFORMANCE:\n"
        f"  Net Sharpe: {baseline_sharpe_net:.3f}\n"
        f"  Ann. Return: {baseline_ann_return:.1%}\n"
        f"  Ann. Vol: {baseline_ann_vol:.1%}\n"
        f"  Max DD: {baseline_max_dd:.1%}\n\n"
        f"IMPROVEMENT:\n"
        f"  Sharpe Improvement: {improvement:.3f}\n"
        f"  Return Improvement: {strategy_ann_return - baseline_ann_return:.1%}\n"
        f"  Vol Improvement: {baseline_ann_vol - strategy_ann_vol:.1%}"
    )
    
    fig.text(0.02, 0.02, metrics_text,
             fontsize=9, verticalalignment='bottom', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if plot_path is None:
        plot_path = f"strategy_results_{title.replace(' ', '_').lower()}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path
