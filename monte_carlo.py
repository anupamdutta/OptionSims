import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from black76 import black76_price

@dataclass
class SimulationConfig:
    opening_sf: float           # Opening Synthetic Future (e.g. 8296.3)
    upside_target: float        # Projected Upside Level (e.g. 8350.0 or user input)
    downside_target: float      # Projected Downside Level (e.g. 8185.8)
    dte_days: float             # Total DTE in days (e.g. 1.621528)
    opening_iv: float           # Opening IV in decimal (e.g. 0.54043 for 54.043%)
    r: float = 0.0              # Risk-free rate (default 0.0)
    call_strike: Optional[float] = None  # If None, defaults to opening_sf
    put_strike: Optional[float] = None   # If None, defaults to opening_sf
    num_simulations: int = 10000         # Number of Monte Carlo paths
    time_steps: int = 500                # Number of time steps
    iv_skew_beta: float = -0.08          # IV sensitivity slider: d(IV) / d(SF/SF0)
    minutes_per_day: float = 385.0       # 9:15 AM to 3:40 PM = 385 minutes

@dataclass
class TargetHitStats:
    target_name: str
    target_level: float
    total_paths: int
    hit_count: int
    hit_probability_pct: float
    avg_elapsed_minutes: float
    avg_elapsed_days: float
    avg_remaining_dte: float
    iv_at_hit_pct: float
    
    # Call Stats
    call_mean: float
    call_median: float
    call_min: float
    call_max: float
    call_p05: float
    call_p95: float
    call_std: float
    
    # Put Stats
    put_mean: float
    put_median: float
    put_min: float
    put_max: float
    put_p05: float
    put_p95: float
    put_std: float
    
    # Raw arrays for histogram / detailed inspection
    hit_call_prices: np.ndarray
    hit_put_prices: np.ndarray
    hit_elapsed_minutes: np.ndarray
    hit_indices: np.ndarray


def run_monte_carlo(cfg: SimulationConfig) -> Dict[str, Any]:
    """
    Executes vectorized Monte Carlo simulation under Black-76 (zero drift martingale).
    Tracks first hitting times of upside and downside targets.
    Computes distribution of Call and Put prices at the moment the target is reached.
    """
    call_k = cfg.call_strike if cfg.call_strike is not None else cfg.opening_sf
    put_k = cfg.put_strike if cfg.put_strike is not None else cfg.opening_sf
    
    N = cfg.num_simulations
    M = cfg.time_steps
    
    # Time discretization (365 calendar days convention)
    dt_days = cfg.dte_days / M
    dt_years = dt_days / 365.0
    dt_minutes = (cfg.dte_days * cfg.minutes_per_day) / M
    
    # Synthetic Future path simulation under Black-76
    # d ln(F) = -0.5 * sigma^2 * dt + sigma * sqrt(dt) * Z
    drift = -0.5 * (cfg.opening_iv ** 2) * dt_years
    diffusion = cfg.opening_iv * np.sqrt(dt_years)
    
    # Generate random standard normals: shape (N, M)
    Z = np.random.standard_normal((N, M))
    log_returns = drift + diffusion * Z
    
    # Cumulative sum along time steps
    cum_log_returns = np.cumsum(log_returns, axis=1)
    
    # Full price paths: shape (N, M + 1)
    # Start at F0
    F_paths = np.empty((N, M + 1), dtype=np.float64)
    F_paths[:, 0] = cfg.opening_sf
    F_paths[:, 1:] = cfg.opening_sf * np.exp(cum_log_returns)
    
    # Time arrays
    elapsed_steps = np.arange(M + 1)
    elapsed_minutes_arr = elapsed_steps * dt_minutes
    elapsed_dte_arr = elapsed_steps * dt_days
    remaining_dte_arr = np.maximum(0.0, cfg.dte_days - elapsed_dte_arr)
    remaining_T_years = remaining_dte_arr / 365.0

    # IV Skew Calculation:
    # dIV = beta * ((F_target - F0) / F0)
    # If beta < 0 (typical equity index skew), downside hit raises IV, upside hit lowers IV
    def get_target_iv(target_price: float) -> float:
        pct_change = (target_price - cfg.opening_sf) / cfg.opening_sf
        delta_iv = cfg.iv_skew_beta * pct_change
        return float(np.clip(cfg.opening_iv + delta_iv, 0.05, 3.0))

    # Helper to analyze hitting stats for a given target level
    def analyze_barrier(target_price: float, name: str, is_upside: bool) -> TargetHitStats:
        if is_upside:
            # Condition: price >= target_price
            hit_mask = F_paths >= target_price
        else:
            # Condition: price <= target_price
            hit_mask = F_paths <= target_price
            
        # Find first hitting step index for each path
        has_hit = np.any(hit_mask, axis=1)
        first_hit_step = np.argmax(hit_mask, axis=1) # Returns first index of True if exists
        
        hit_indices = np.where(has_hit)[0]
        hit_count = len(hit_indices)
        hit_prob = (hit_count / N) * 100.0
        
        iv_at_hit = get_target_iv(target_price)
        
        if hit_count > 0:
            hit_steps = first_hit_step[hit_indices]
            hit_rem_T = remaining_T_years[hit_steps]
            hit_rem_dte = remaining_dte_arr[hit_steps]
            hit_el_mins = elapsed_minutes_arr[hit_steps]
            hit_el_days = elapsed_dte_arr[hit_steps]
            
            # Black-76 option pricing at the exact target level
            call_prices = black76_price(target_price, call_k, hit_rem_T, cfg.r, iv_at_hit, 'call')
            put_prices = black76_price(target_price, put_k, hit_rem_T, cfg.r, iv_at_hit, 'put')
            
            stats = TargetHitStats(
                target_name=name,
                target_level=target_price,
                total_paths=N,
                hit_count=hit_count,
                hit_probability_pct=hit_prob,
                avg_elapsed_minutes=float(np.mean(hit_el_mins)),
                avg_elapsed_days=float(np.mean(hit_el_days)),
                avg_remaining_dte=float(np.mean(hit_rem_dte)),
                iv_at_hit_pct=iv_at_hit * 100.0,
                
                call_mean=float(np.mean(call_prices)),
                call_median=float(np.median(call_prices)),
                call_min=float(np.min(call_prices)),
                call_max=float(np.max(call_prices)),
                call_p05=float(np.percentile(call_prices, 5)),
                call_p95=float(np.percentile(call_prices, 95)),
                call_std=float(np.std(call_prices)),
                
                put_mean=float(np.mean(put_prices)),
                put_median=float(np.median(put_prices)),
                put_min=float(np.min(put_prices)),
                put_max=float(np.max(put_prices)),
                put_p05=float(np.percentile(put_prices, 5)),
                put_p95=float(np.percentile(put_prices, 95)),
                put_std=float(np.std(put_prices)),
                
                hit_call_prices=call_prices,
                hit_put_prices=put_prices,
                hit_elapsed_minutes=hit_el_mins,
                hit_indices=hit_indices
            )
        else:
            empty_arr = np.array([], dtype=float)
            stats = TargetHitStats(
                target_name=name,
                target_level=target_price,
                total_paths=N,
                hit_count=0,
                hit_probability_pct=0.0,
                avg_elapsed_minutes=0.0,
                avg_elapsed_days=0.0,
                avg_remaining_dte=0.0,
                iv_at_hit_pct=iv_at_hit * 100.0,
                call_mean=0.0, call_median=0.0, call_min=0.0, call_max=0.0, call_p05=0.0, call_p95=0.0, call_std=0.0,
                put_mean=0.0, put_median=0.0, put_min=0.0, put_max=0.0, put_p05=0.0, put_p95=0.0, put_std=0.0,
                hit_call_prices=empty_arr, hit_put_prices=empty_arr, hit_elapsed_minutes=empty_arr, hit_indices=empty_arr
            )
        return stats

    # Upside target analysis
    upside_stats = analyze_barrier(cfg.upside_target, "Upside Target", is_upside=True)
    # Downside target analysis
    downside_stats = analyze_barrier(cfg.downside_target, "Downside Target", is_upside=False)
    
    # ── SF Path-level High / Low statistics across ALL N paths ──────────────
    # For each path, the highest and lowest SF value reached during the session
    path_highs = F_paths.max(axis=1)   # shape (N,) — peak SF of each path
    path_lows  = F_paths.min(axis=1)   # shape (N,) — trough SF of each path

    sf_path_stats = {
        # Projected HIGH stats
        'high_mean':   float(np.mean(path_highs)),
        'high_median': float(np.median(path_highs)),
        'high_best':   float(np.max(path_highs)),    # most bullish path
        'high_worst':  float(np.min(path_highs)),    # least bullish high
        'high_p05':    float(np.percentile(path_highs, 5)),
        'high_p95':    float(np.percentile(path_highs, 95)),
        # Projected LOW stats
        'low_mean':    float(np.mean(path_lows)),
        'low_median':  float(np.median(path_lows)),
        'low_best':    float(np.min(path_lows)),     # most bearish path
        'low_worst':   float(np.max(path_lows)),     # least bearish low
        'low_p05':     float(np.percentile(path_lows, 5)),
        'low_p95':     float(np.percentile(path_lows, 95)),
    }

    # Opening baseline Black-76 prices (at t=0, F = F0, T = DTE / 365)
    T_init = cfg.dte_days / 365.0
    opening_call = float(black76_price(cfg.opening_sf, call_k, T_init, cfg.r, cfg.opening_iv, 'call'))
    opening_put = float(black76_price(cfg.opening_sf, put_k, T_init, cfg.r, cfg.opening_iv, 'put'))
    
    return {
        'config': cfg,
        'opening_call': opening_call,
        'opening_put': opening_put,
        'call_strike': call_k,
        'put_strike': put_k,
        'upside_stats': upside_stats,
        'downside_stats': downside_stats,
        'sf_path_stats': sf_path_stats,
        'paths_sample': F_paths[:60], # 60 paths for charting to keep GUI rendering swift
        'time_axis_minutes': elapsed_minutes_arr
    }

if __name__ == '__main__':
    # Test configuration
    # Note: user mentioned: opening SF 8296.3, projected upside 8347.3 (or upside level), downside 8185.8
    cfg = SimulationConfig(
        opening_sf=8296.3,
        upside_target=8350.0,
        downside_target=8185.8,
        dte_days=1.621528,
        opening_iv=0.54043,
        r=0.0,
        call_strike=8300.0,
        put_strike=8300.0,
        num_simulations=10000,
        time_steps=385,
        iv_skew_beta=-0.35
    )
    res = run_monte_carlo(cfg)
    up = res['upside_stats']
    down = res['downside_stats']
    print(f"Opening Baseline Call: {res['opening_call']:.2f}, Put: {res['opening_put']:.2f}")
    print(f"Upside Hit Prob: {up.hit_probability_pct:.1f}%, Avg Call: {up.call_mean:.2f}, Best Call: {up.call_max:.2f}, Avg Put: {up.put_mean:.2f}")
    print(f"Downside Hit Prob: {down.hit_probability_pct:.1f}%, Avg Call: {down.call_mean:.2f}, Avg Put: {down.put_mean:.2f}, Best Put: {down.put_max:.2f}")
