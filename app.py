
import streamlit as st

# --- DEPENDENCY SAFETY CHECK ---
# This prevents the app from crashing if the server is missing requirements.txt
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.integrate import solve_ivp
except ModuleNotFoundError as e:
    st.set_page_config(page_title="Deployment Error", layout="centered")
    st.error("🚨 **Deployment Configuration Missing!**")
    st.warning(f"The server is trying to run the app, but it is missing: `{e.name}`")
    st.markdown("""
    ### How to fix this in 30 seconds:
    Streamlit Cloud needs a blueprint to install your math libraries. You are missing a file called `requirements.txt`.
    
    1. Go to your **GitHub repository** where this `app.py` file is saved.
    2. Click **Add file** > **Create new file**.
    3. Name the file exactly: **`requirements.txt`** (all lowercase).
    4. Paste these exact 4 lines into the text box:
    ```text
    streamlit==1.32.0
    numpy>=1.24.0
    scipy>=1.10.0
    matplotlib>=3.7.0
    ```
    5. Click the green **Commit changes** button. 
    
    *Once you do this, Streamlit will automatically restart, install the libraries, and your trader dashboard will appear!*
    """)
    st.stop()  # Stops the rest of the script from running and crashing

# --- Page Configuration ---
st.set_page_config(page_title="Momentum Asset Pricing Model", layout="wide")

# --- Mathematical Model ---
def ode_rhs(t, y, alpha, beta, P_fund, lambda_s, lambda_l, gamma, delta, kappa, eta):
    P, S, L, M = y
    # System of ODEs
    dP = alpha * M * (S - L) + beta * (P_fund - P)
    dS = lambda_s * (P - S)
    dL = lambda_l * (P - L)
    dM = -gamma * M + delta * np.tanh(kappa * (S - L)) - eta * (P - S)
    return np.array([dP, dS, dL, dM])

# --- UI Layout ---
st.title("Dynamical System of Asset Pricing with Momentum Traders")
st.markdown("Explore how momentum trading, moving average crossovers, and market sentiment interact to create price bubbles and crashes.")

# --- Sidebar for Inputs ---
st.sidebar.header("Model Parameters")
st.sidebar.markdown("Adjust the parameters to simulate different market conditions.")

alpha = st.sidebar.slider("Trading Intensity (α)", 0.0, 5.0, 2.5, 0.1, help="How strongly traders react to sentiment and crossover signals.")
beta = st.sidebar.slider("Mean Reversion (β)", 0.0, 2.0, 0.8, 0.1, help="Strength of fundamental value anchoring.")
P_fund = st.sidebar.number_input("Fundamental Price (P*)", value=100.0)

st.sidebar.subheader("Moving Averages")
lambda_s = st.sidebar.slider("Short MA Decay (λ_s)", 0.1, 10.0, 5.0, 0.1, help="Higher = faster reaction. τ_s = 1/λ_s")
lambda_l = st.sidebar.slider("Long MA Decay (λ_l)", 0.1, 5.0, 0.5, 0.1, help="Lower = slower reaction. τ_l = 1/λ_l")

st.sidebar.subheader("Sentiment Dynamics")
gamma = st.sidebar.slider("Sentiment Reversion (γ)", 0.1, 5.0, 1.5, 0.1, help="How fast market hype fades naturally.")
delta = st.sidebar.slider("Crossover Signal Strength (δ)", 0.0, 5.0, 3.0, 0.1)
kappa = st.sidebar.slider("Signal Steepness (κ)", 0.1, 2.0, 0.8, 0.1)
eta = st.sidebar.slider("Price Feedback (η)", 0.0, 0.5, 0.05, 0.01, help="Damping effect when price deviates from the short trend.")

st.sidebar.subheader("Initial Conditions")
P0 = st.sidebar.number_input("Initial Price (P₀)", value=105.0)
S0 = st.sidebar.number_input("Initial Short MA (S₀)", value=100.0)
L0 = st.sidebar.number_input("Initial Long MA (L₀)", value=100.0)
M0 = st.sidebar.number_input("Initial Sentiment (M₀)", value=0.5)
T_sim = st.sidebar.slider("Simulation Horizon (T)", 5, 50, 20)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Market Simulation", "📖 Parameter Guide", "📈 How to Read Results"])

with tab1:
    st.subheader("Forward Trajectory Simulation")
    
    # Run Simulation
    y0 = [P0, S0, L0, M0]
    t_eval = np.linspace(0, T_sim, 1000)
    
    try:
        sol = solve_ivp(
            ode_rhs, (0, T_sim), y0, 
            args=(alpha, beta, P_fund, lambda_s, lambda_l, gamma, delta, kappa, eta),
            method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-11
        )
        
        t = sol.t
        P, S, L, M = sol.y
        cross = S - L

        # Plotting Setup
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3)

        # Panel 1: Price Plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, P, color='#1f4e79', lw=2, label='Price P(t)')
        ax1.plot(t, S, color='#2e75b6', lw=1.5, ls='--', label='Short MA S(t)')
        ax1.plot(t, L, color='#ed7d31', lw=1.5, ls='-.', label='Long MA L(t)')
        ax1.axhline(P_fund, color='#c00000', lw=1.2, ls=':', label='P* (Fundamental)')
        ax1.set_title('Asset Price Dynamics')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # Panel 2: Sentiment Plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(t, M, color='#70ad47', lw=1.8)
        ax2.axhline(0, color='black', lw=0.8, ls='--')
        ax2.fill_between(t, M, 0, where=(M > 0), alpha=0.3, color='#70ad47', label='Bullish')
        ax2.fill_between(t, M, 0, where=(M < 0), alpha=0.3, color='#c00000', label='Bearish')
        ax2.set_title('Market Sentiment M(t)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Sentiment')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        # Panel 3: Phase Portrait
        ax3 = fig.add_subplot(gs[1, 1])
        sc = ax3.scatter(P, M, c=t, cmap='plasma', s=10)
        ax3.plot(P_fund, 0, 'k*', ms=12, label='Equilibrium')
        plt.colorbar(sc, ax=ax3, label='Time')
        ax3.set_title('Phase Portrait: Price vs Sentiment')
        ax3.set_xlabel('Price')
        ax3.set_ylabel('Sentiment')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()

        # Render Safely
        st.pyplot(fig, clear_figure=True)
        plt.close(fig) # Closes the figure to prevent memory faults
        
    except Exception as e:
        st.error(f"Simulation failed due to numerical instability: {e}. Try lowering the trading intensity (α) or adjusting initial conditions.")

with tab2:
    st.markdown("""
    ### 📖 Where to Find and Estimate Parameters
    This model is a continuous-time dynamical system. You cannot pull these parameters directly from an API like Yahoo Finance. Instead, they must be estimated heuristically or via regression on historical data.

    * **$P^*$ (Fundamental Price):** Can be estimated using Discounted Cash Flow (DCF) models, Book Value, or a very long-term moving average (e.g., 200-day MA) depending on the asset.
    * **$\lambda_s$ and $\lambda_l$ (Moving Average Decay):** These are the inverse of the time horizons. If you use a 10-day short EMA and a 50-day long EMA, then $\lambda_s \approx 1/10$ and $\lambda_l \approx 1/50$. Adjust based on your strategy's timeframe.
    * **$\alpha$ (Trading Intensity):** Represents market liquidity and volume. High volume, highly speculative assets (like certain crypto tokens or meme stocks) will have a high $\alpha$. Blue-chip stocks have a lower $\alpha$.
    * **$\beta$ (Mean Reversion):** Represents value investors or market makers. A high $\beta$ means the asset quickly corrects to its fair value.
    * **$\gamma$ (Sentiment Reversion):** Represents the news cycle decay. How quickly do traders forget the hype? Usually estimated between 1.0 (slow memory decay) and 5.0 (rapid news cycle).
    """)

with tab3:
    st.markdown("""
    ### 📈 How to Understand the Results

    **1. Asset Price Dynamics (Top Chart)**
    * Watch the interaction between the solid blue line (Price) and the dotted red line (Fundamental). 
    * When the Short MA (dashed blue) crosses above the Long MA (dashed orange), it acts as a "Golden Cross," generating buy signals for momentum traders.

    **2. Market Sentiment (Bottom Left)**
    * **Green areas** indicate irrational exuberance (bullish hype).
    * **Red areas** indicate panic selling (bearish fear).
    * If Sentiment stays high while the price is far above $P^*$, you are witnessing a **bubble**. The steeper the drop from green to red, the harder the crash.

    **3. Phase Portrait (Bottom Right)**
    * This chart maps the exact relationship between Price and Sentiment over time (color-coded from dark to bright yellow as time passes).
    * A spiral converging on the black star means the market stabilizes.
    * A wide, outward-spinning circle indicates a volatile, unmoored market.
    """)
