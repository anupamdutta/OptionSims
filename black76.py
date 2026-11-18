import numpy as np
from scipy.stats import norm

def black76_price(F, K, T, r, sigma, option_type='call'):
    """
    Vectorized Black-76 option pricing formula for European options on futures/synthetic futures.
    F: Forward / Synthetic Future price
    K: Strike price
    T: Time to expiration in years (DTE / 365)
    r: Risk-free interest rate (e.g. 0.0)
    sigma: Implied volatility (e.g. 0.54043 for 54.043%)
    option_type: 'call' or 'put'
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Intrinsic value when T <= 0 or sigma <= 0
    df = np.exp(-r * np.maximum(T, 0.0))
    
    is_expired = (T <= 1e-7) | (sigma <= 1e-7)
    
    # Avoid div by zero in normal calculation
    safe_T = np.where(is_expired, 1e-6, T)
    safe_sigma = np.where(is_expired, 1e-6, np.maximum(sigma, 1e-6))
    
    sqrt_T = np.sqrt(safe_T)
    d1 = (np.log(F / K) + 0.5 * safe_sigma**2 * safe_T) / (safe_sigma * sqrt_T)
    d2 = d1 - safe_sigma * sqrt_T
    
    if option_type.lower() == 'call':
        bs_price = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
        intrinsic = np.maximum(0.0, F - K)
    else:
        bs_price = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        intrinsic = np.maximum(0.0, K - F)
        
    price = np.where(is_expired, intrinsic, bs_price)
    return np.maximum(0.0, price)

if __name__ == '__main__':
    # Test with user inputs:
    # SF = 8296.3, DTE = 1.621528 days -> T = 1.621528 / 365, r = 0, IV = 54.043%
    sf = 8296.3
    dte = 1.621528
    T = dte / 365.0
    r = 0.0
    iv = 0.54043
    call_atm = black76_price(sf, sf, T, r, iv, 'call')
    put_atm = black76_price(sf, sf, T, r, iv, 'put')
    print(f"Opening SF: {sf}")
    print(f"ATM Call: {call_atm:.2f}, ATM Put: {put_atm:.2f}")
    assert np.isclose(call_atm, put_atm), "At r=0 and F=K, Call must equal Put under Black-76!"
    print("Black-76 unit check passed successfully!")
