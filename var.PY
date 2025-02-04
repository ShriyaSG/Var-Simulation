import numpy as np
import matplotlib.pyplot as plt

def simulate_var(model_params, num_simulations=5000, confidence_level=0.99):
    # Extract parameters
    S = model_params['S']
    P = model_params['P']
    delta = model_params['delta']
    gamma = model_params['gamma']
    sigma = model_params['sigma']
    T = model_params['T']
    
    # Input validation
    if sigma <= 0 or T <= 0:
        raise ValueError("Volatility (sigma) and time horizon (T) must be positive.")
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    
    # Simulate changes in the underlying asset price (Δx)
    delta_x = np.random.normal(0, sigma * np.sqrt(T), num_simulations)
    
    # Compute portfolio value changes (ΔP) using the quadratic model
    delta_P = S * delta * delta_x + 0.5 * S**2 * gamma * delta_x**2
    
    # Sort ΔP values to determine the left tail
    sorted_losses = np.sort(delta_P)
    
    # Determine the index for VaR
    var_index = int((1 - confidence_level) * num_simulations)
    
    # VaR is the value at the confidence level
    VaR = sorted_losses[var_index]
    
    # Normalize VaR as a percentage of the portfolio value
    VaR_percentage = VaR / P * 100
    
    # Plot histogram with shaded area for VaR
    plt.figure(figsize=(12, 6))
    plt.hist(delta_P, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Portfolio ΔP Distribution')
    plt.axvline(VaR, color='red', linestyle='--', linewidth=2, label=f'VaR (99%): {VaR:.2f}')
    plt.fill_betweenx([0, plt.gca().get_ylim()[1]], sorted_losses[0], VaR, color='red', alpha=0.3, label='VaR Region')
    plt.title("Monte Carlo Simulation: Portfolio Value Changes (ΔP)", fontsize=14)
    plt.xlabel("Change in Portfolio Value (ΔP)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.4)
    plt.show()
    
    # Plot cumulative distribution function (CDF)
    plt.figure(figsize=(12, 6))
    cdf = np.arange(1, num_simulations + 1) / num_simulations
    plt.plot(sorted_losses, cdf, label='Cumulative Distribution', color='blue')
    plt.axvline(VaR, color='red', linestyle='--', label=f'VaR (99%): {VaR:.2f}')
    plt.axhline(1 - confidence_level, color='green', linestyle='--', label=f'Confidence Level: {confidence_level * 100:.1f}%')
    plt.title("Cumulative Distribution Function (CDF) of ΔP", fontsize=14)
    plt.xlabel("Change in Portfolio Value (ΔP)", fontsize=12)
    plt.ylabel("Cumulative Probability", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.4)
    plt.show()
    
    return VaR, VaR_percentage

if __name__ == "__main__":
    # Define portfolio parameters
    portfolio_params = {
        'S': 100,           # Current price of the underlying asset
        'P': 1000000,       # Current portfolio value
        'delta': 0.5,       # Portfolio delta
        'gamma': 0.01,      # Portfolio gamma
        'sigma': 0.02,      # Daily volatility (2%)
        'T': 3              # Time horizon (3 days)
    }
    var_99, var_percentage = simulate_var(portfolio_params, num_simulations=5000, confidence_level=0.99)
    print(f"1-day 99% VaR: {var_99:.2f}")
