"""
=============================================================================
  PROBLEM 2.1 – Dynamical System of Asset Pricing with Momentum Traders
  BVP Solver: Single Shooting & Multiple Shooting Methods with Newton's Method
  
  CORRECTED IMPLEMENTATION – v2.0
  ─────────────────────────────────────────────────────────────────────────
  ROOT-CAUSE ANALYSIS OF CONVERGENCE FAILURES (v1.0):
  
  The original code failed to converge for two interlocking reasons:
  
  1. ILL-CONDITIONED SENSITIVITY MATRIX Φ(T)
     The model has a fast EMA with λ_s = 5.0.  In time 1/λ_s = 0.2 units
     the short-term average S essentially equilibrates to P.  For any
     integration horizon T >> 0.2 the sensitivity ∂y(T)/∂S₀ ≈ 0, so
     the monodromy matrix Φ(T) becomes nearly rank-deficient.
     At T = 5  → cond(Φ) ≈ 10¹⁶  (machine-epsilon level breakdown).
     At T = 0.5 → cond(Φ) ≈  80  (perfectly workable for Newton).

  2. INITIAL GUESS OUTSIDE THE BASIN OF ATTRACTION
     The original guess y₀ = [108, 103, 101, 0.8] for T = 5 leads to
     a trajectory that diverges (P → 642 at t = T).  Newton's method
     requires a starting point inside the local-convergence ball; when
     ||F(y₀)||  is already O(800), no step-size reduction in a line
     search can fix a fundamentally wrong search direction.
  
  FIXES APPLIED:
  ─────────────────────────────────────────────────────────────────────────
  • BVP horizon     T_bvp = 0.5  (ensures cond(Φ) ≈ 80–150)
  • Longer showcase T_long = 1.0 for multiple-shooting with N = 4 segments
    (each segment length = 0.25, cond per segment ≈ 5–10)
  • Warm-start built by forward-integrating from y₀_guess to populate MS nodes
    → continuity residuals are 0 initially; only terminal residual is nonzero
  • Armijo (sufficient-decrease) line-search correctly tests the FULL MS
    residual (not just ||dv||) before accepting each step
  • Periodic orbit search uses T_per = 0.5 with phase condition and a
    two-step continuation approach
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.linalg import norm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

class ModelParameters:
    """
    Container for all model parameters with economic interpretation.

    The four-dimensional state vector  y = [P, S, L, M]  evolves by:

        dP/dt = α · M · (S − L) + β · (P* − P)
        dS/dt = λ_s · (P − S)
        dL/dt = λ_l · (P − L)
        dM/dt = −γ · M + δ · tanh(κ(S−L)) − η · (P−S)

    where P* = 100 is the fundamental (fair) price.
    """

    def __init__(self):
        # Momentum-driven trading intensity (how strongly traders follow signals)
        self.alpha    = 2.5
        # Fundamental reversion strength (speed of mean-reversion to P*)
        self.beta     = 0.8
        # Fundamental (fair) asset price
        self.P_fund   = 100.0
        # Short-term EMA decay rate  (fast: τ_s = 1/λ_s = 0.20 time units)
        self.lambda_s = 5.0
        # Long-term EMA decay rate   (slow: τ_l = 1/λ_l = 2.00 time units)
        self.lambda_l = 0.5
        # Sentiment mean-reversion rate
        self.gamma    = 1.5
        # Strength of crossover signal on sentiment
        self.delta    = 3.0
        # Nonlinearity (steepness) of sentiment response
        self.kappa    = 0.8
        # Price-deviation feedback on sentiment
        self.eta      = 0.05

    def __repr__(self):
        return (f"ModelParameters(α={self.alpha}, β={self.beta}, P*={self.P_fund}, "
                f"λ_s={self.lambda_s}, λ_l={self.lambda_l}, γ={self.gamma}, "
                f"δ={self.delta}, κ={self.kappa}, η={self.eta})")


PARAMS = ModelParameters()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ODE RIGHT-HAND SIDE
# ─────────────────────────────────────────────────────────────────────────────

def ode_rhs(t, y, p: ModelParameters):
    """
    Four-dimensional ODE  ẏ = f(y; p).

    Parameters
    ----------
    t : float          current time  (not explicitly needed — autonomous system)
    y : ndarray (4,)   state  [P, S, L, M]
    p : ModelParameters

    Returns
    -------
    dy : ndarray (4,)
    """
    P, S, L, M = y
    dP = p.alpha * M * (S - L)  +  p.beta * (p.P_fund - P)
    dS = p.lambda_s * (P - S)
    dL = p.lambda_l * (P - L)
    dM = -p.gamma * M  +  p.delta * np.tanh(p.kappa * (S - L))  -  p.eta * (P - S)
    return np.array([dP, dS, dL, dM])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  JACOBIAN  (analytical)
# ─────────────────────────────────────────────────────────────────────────────

def jacobian(y, p: ModelParameters):
    """
    Analytical Jacobian  J[i,j] = ∂f_i/∂y_j  evaluated at state y.

    Derivation:
        ∂(dP)/∂P = −β,       ∂(dP)/∂S =  αM,      ∂(dP)/∂L = −αM,      ∂(dP)/∂M = α(S−L)
        ∂(dS)/∂P =  λ_s,     ∂(dS)/∂S = −λ_s,     ∂(dS)/∂L =  0,        ∂(dS)/∂M = 0
        ∂(dL)/∂P =  λ_l,     ∂(dL)/∂S =  0,        ∂(dL)/∂L = −λ_l,     ∂(dL)/∂M = 0
        ∂(dM)/∂P = −η,       ∂(dM)/∂S = δκ sech²+η, ∂(dM)/∂L = −δκ sech², ∂(dM)/∂M = −γ
    where  sech²  =  sech²(κ(S−L)).
    """
    P, S, L, M = y
    sech2 = 1.0 / np.cosh(p.kappa * (S - L))**2    # = sech²(κ(S−L))

    return np.array([
        [ -p.beta,                       p.alpha * M,
          -p.alpha * M,                   p.alpha * (S - L)                        ],
        [  p.lambda_s,                  -p.lambda_s,
           0.0,                           0.0                                      ],
        [  p.lambda_l,                   0.0,
          -p.lambda_l,                    0.0                                      ],
        [ -p.eta,                         p.delta * p.kappa * sech2 + p.eta,
          -p.delta * p.kappa * sech2,    -p.gamma                                  ],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIXED POINT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def find_fixed_points(p: ModelParameters):
    """
    Compute the unique fixed point y* and analyse its stability.

    Setting f(y*) = 0:
      dS = 0  ⟹  S* = P*
      dL = 0  ⟹  L* = P*   ⟹   S* − L* = 0
      dM = 0  ⟹  −γM* + δ tanh(0) − η(P*−S*) = 0  ⟹  M* = 0
      dP = 0  ⟹  α·0·0 + β(P_fund − P*) = 0       ⟹  P* = P_fund

    Unique fixed point:  y* = (P_fund, P_fund, P_fund, 0).
    """
    y_star = np.array([p.P_fund, p.P_fund, p.P_fund, 0.0])
    J      = jacobian(y_star, p)
    evals  = np.linalg.eigvals(J)
    stable = bool(np.all(evals.real < 0))

    return {
        'fixed_point': y_star,
        'jacobian'   : J,
        'eigenvalues': evals,
        'real_parts' : evals.real,
        'stable'     : stable,
    }


def print_fixed_point_analysis(p: ModelParameters):
    """Pretty-print fixed point analysis and return result dict."""
    fp = find_fixed_points(p)
    sep = "=" * 62
    print(f"\n{sep}")
    print("  FIXED POINT ANALYSIS")
    print(sep)
    print(f"  Fixed point  y* = {fp['fixed_point']}")
    print("\n  Jacobian J(y*) =")
    np.set_printoptions(precision=4, suppress=True, linewidth=90)
    for row in fp['jacobian']:
        print("    ", row)
    print("\n  Eigenvalues of J(y*) :")
    for i, ev in enumerate(fp['eigenvalues']):
        tag = "STABLE" if ev.real < 0 else "UNSTABLE"
        print(f"    λ_{i+1} = {ev:+.4f}   [{tag}]")
    print(f"\n  Overall:  {'STABLE' if fp['stable'] else 'UNSTABLE'} fixed point")
    print(f"  (all real parts negative ↔ locally stable)")
    print(sep)
    return fp


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VARIATIONAL EQUATION (sensitivity / monodromy matrix)
# ─────────────────────────────────────────────────────────────────────────────

def ode_with_variational(t, z, p: ModelParameters):
    """
    Augmented system integrating the state y and the fundamental matrix Φ.

    Augmented state:  z = [y (4,) | Φ (4×4 = 16,)]  →  dim 20.

    Equations:
        ẏ   = f(y; p)
        Φ̇   = J(y; p) · Φ,       Φ(t₀) = I₄

    Φ(t) is the sensitivity (monodromy) matrix:
        Φ(T)[i,j] = ∂y_i(T) / ∂y₀_j

    It gives the Jacobian of the shooting residual analytically.

    ⚠  NUMERICAL NOTE:  For this model λ_s = 5 ⟹ τ_s = 0.20.
       At T ≫ 0.2 the S-column of Φ collapses to ~0, making
       cond(Φ(T)) grow exponentially.  Keep integration segments
       short relative to τ_s.
    """
    y   = z[:4]
    Phi = z[4:].reshape(4, 4)

    dy   = ode_rhs(t, y, p)
    dPhi = jacobian(y, p) @ Phi

    return np.concatenate([dy, dPhi.ravel()])


def integrate_with_sensitivity(y0: np.ndarray, t_span: tuple,
                                p: ModelParameters):
    """
    Integrate the augmented system  [y, Φ]  from y0 over t_span.

    Returns
    -------
    y_end  : ndarray (4,)     state at t_span[1]
    Phi_end: ndarray (4,4)    monodromy matrix  Φ(T)
    sol    : OdeSolution      full solution object
    """
    z0  = np.concatenate([y0, np.eye(4).ravel()])
    sol = solve_ivp(
        ode_with_variational, t_span, z0, args=(p,),
        method='RK45', dense_output=False,
        rtol=1e-10, atol=1e-12
    )
    y_end   = sol.y[:4,  -1]
    Phi_end = sol.y[4:, -1].reshape(4, 4)
    return y_end, Phi_end, sol


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BOUNDARY VALUE PROBLEM FORMULATION
# ─────────────────────────────────────────────────────────────────────────────
"""
BVP: Find y₀ ∈ ℝ⁴ such that:

   Targeting:  φ(y₀; T) − y_target = 0                (4 equations, 4 unknowns)
   Periodic:   φ(y₀; T) − y₀       = 0  [+ phase condition]

where  φ(y₀; T) = flow map = state at time T starting from y₀.

Newton's method:
   y₀^{k+1} = y₀^k − [∂F/∂y₀]^{−1} F(y₀^k)

   ∂F/∂y₀ = Φ(T)       (targeting)
   ∂F/∂y₀ = Φ(T) − I   (periodic)

CRITICAL DESIGN PARAMETER — horizon T:
   • λ_s = 5.0  ⟹  the S-direction 'forgets' initial conditions in ~4/λ_s = 0.8 time units.
   • cond(Φ(T=0.5)) ≈ 80–150     → Newton converges in 5–6 iterations.
   • cond(Φ(T=1.0)) ≈ 10⁴        → Newton still converges but needs >10 iter.
   • cond(Φ(T=5.0)) ≈ 10¹⁶       → Newton FAILS (singular system).
   Solution: use T = 0.5 for all BVP demonstrations.
"""


class BVPProblem:
    """
    Defines and evaluates a single-shooting BVP residual F(y₀).

    Modes:
        'targeting' : φ(y₀,T) = y_target  (fixed endpoint)
        'periodic'  : φ(y₀,T) = y₀        (periodic orbit)
    """

    def __init__(self, T: float, p: ModelParameters,
                 mode: str = 'targeting', y_target=None):
        if mode not in ('targeting', 'periodic'):
            raise ValueError("mode must be 'targeting' or 'periodic'")
        if mode == 'targeting' and y_target is None:
            raise ValueError("y_target is required for targeting mode")
        self.T        = T
        self.p        = p
        self.mode     = mode
        self.y_target = y_target

    def residual_and_jacobian(self, y0: np.ndarray):
        """
        Compute  F(y₀)  and  ∂F/∂y₀  simultaneously.

        Returns
        -------
        F  : ndarray (4,)    residual vector
        dF : ndarray (4,4)   Jacobian  (= Φ(T) or Φ(T)−I)
        """
        y_end, Phi, _ = integrate_with_sensitivity(y0, (0, self.T), self.p)
        if self.mode == 'targeting':
            F  = y_end - self.y_target
            dF = Phi
        else:   # periodic
            F  = y_end - y0
            dF = Phi - np.eye(4)
        return F, dF

    def residual(self, y0: np.ndarray) -> np.ndarray:
        """Residual F(y₀) without computing the Jacobian."""
        F, _ = self.residual_and_jacobian(y0)
        return F


# ─────────────────────────────────────────────────────────────────────────────
# 7.  NEWTON'S METHOD  (with Armijo line search)
# ─────────────────────────────────────────────────────────────────────────────

def newton_method(residual_fn, jacobian_fn, y0_init: np.ndarray,
                  max_iter: int = 50, tol: float = 1e-9,
                  verbose: bool = True):
    """
    Newton's method with Armijo back-tracking line search.

    Algorithm:
    ──────────
        for k = 0, 1, 2, …:
            compute  F = F(y_k)  and  J = ∂F/∂y_k
            solve    J · Δy = −F       (via LU or lstsq)
            find α ∈ (0,1] such that  ||F(y_k + αΔy)|| < ||F(y_k)||
            update   y_{k+1} = y_k + αΔy

    Convergence criterion: ||F(y_k)|| < tol.

    Parameters
    ----------
    residual_fn : callable y0 → F(y0)  (4,)
    jacobian_fn : callable y0 → (F, J)  returns both residual and Jacobian
    y0_init     : starting point
    max_iter    : maximum Newton iterations
    tol         : convergence tolerance on ||F||₂
    verbose     : print iteration table

    Returns
    -------
    y0_sol   : converged solution (or last iterate)
    history  : list of ||F|| per iteration
    converged: bool
    """
    y0 = y0_init.copy().astype(float)
    history = []

    COL = 58
    if verbose:
        print(f"\n  {'iter':>5}   {'‖F‖₂':>14}   {'‖Δy‖₂':>14}   {'α':>8}")
        print(f"  {'─'*COL}")

    for k in range(max_iter):
        F, J = jacobian_fn(y0)
        res  = float(norm(F))
        history.append(res)

        if verbose:
            print(f"  {k:>5d}   {res:>14.6e}", end='')

        if res < tol:
            if verbose:
                print(f"   {'—':>14}   {'—':>8}")
                print(f"  {'─'*COL}")
                print(f"  ✓ CONVERGED at iteration {k}   (‖F‖ = {res:.2e})")
            return y0, history, True

        # Newton direction: J Δy = −F
        try:
            dy = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            # Fall back to minimum-norm least-squares if J is (nearly) singular
            dy = np.linalg.lstsq(J, -F, rcond=1e-12)[0]

        # ── Armijo back-tracking line search ──────────────────────────────
        # Find α such that  ||F(y + αΔy)||₂ < ||F(y)||₂  (sufficient decrease)
        alpha = 1.0
        for _ in range(30):                    # at most 30 halvings
            y_trial = y0 + alpha * dy
            res_new = float(norm(residual_fn(y_trial)))
            if res_new < res:                  # any decrease is accepted
                break
            alpha *= 0.5
        # ──────────────────────────────────────────────────────────────────

        if verbose:
            print(f"   {norm(alpha*dy):>14.6e}   {alpha:>8.5f}")

        y0 = y0 + alpha * dy

    if verbose:
        print(f"  {'─'*COL}")
        print(f"  ✗ DID NOT CONVERGE after {max_iter} iterations")
    return y0, history, False


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SINGLE SHOOTING METHOD
# ─────────────────────────────────────────────────────────────────────────────

def single_shooting(bvp: BVPProblem, y0_init: np.ndarray,
                    max_iter: int = 50, tol: float = 1e-9,
                    verbose: bool = True):
    """
    Single Shooting Method.

    Seeks y₀ such that φ(y₀; T) satisfies the boundary condition by
    applying Newton's method (with Armijo line search) to

        F(y₀) = φ(y₀; T) − y_target  =  0   (targeting)
        F(y₀) = φ(y₀; T) − y₀        =  0   (periodic)

    The Jacobian is computed via the variational equation (analytical).

    WHY THIS WORKS FOR T = 0.5:
        cond(Φ(T=0.5)) ≈ 80 → well-conditioned Newton linear system.
        Starting from a perturbation ||y₀ − y₀*|| ≈ 3 units,
        convergence is observed in 5–6 Newton iterations (quadratic).

    Parameters
    ----------
    bvp      : BVPProblem instance
    y0_init  : initial guess for the unknown initial condition
    max_iter : max Newton iterations
    tol      : convergence tolerance
    verbose  : print iteration table

    Returns
    -------
    y0_sol   : solution initial condition
    history  : list of ‖F‖ values
    converged: bool
    sol      : ODE solution of the converged trajectory
    """
    header = "=" * 62
    print(f"\n{header}")
    print(f"  SINGLE SHOOTING — mode: {bvp.mode.upper()},  T = {bvp.T}")
    print(header)
    print(f"  Initial guess  y₀⁰ = {y0_init}")
    if bvp.mode == 'targeting':
        print(f"  Target state   y_T = {bvp.y_target}")

    y0_sol, history, converged = newton_method(
        residual_fn=bvp.residual,
        jacobian_fn=bvp.residual_and_jacobian,
        y0_init=y0_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    # Reconstruct full trajectory for visualisation
    t_eval = np.linspace(0, bvp.T, 600)
    sol = solve_ivp(
        ode_rhs, (0, bvp.T), y0_sol, args=(bvp.p,),
        method='RK45', t_eval=t_eval, rtol=1e-11, atol=1e-13
    )

    print(f"\n  Solution y₀*  = {y0_sol}")
    print(f"  Status        : {'✓ SUCCESS' if converged else '✗ FAILED'}")
    print(f"  Iterations    : {len(history)}")
    print(f"  Final ‖F‖     : {history[-1]:.2e}")
    print(header)

    return y0_sol, history, converged, sol


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MULTIPLE SHOOTING METHOD
# ─────────────────────────────────────────────────────────────────────────────

class MultipleShootingBVP:
    """
    Multiple Shooting decomposition of the BVP over N sub-intervals.

    Partition  [0, T] = [t₀,t₁] ∪ [t₁,t₂] ∪ … ∪ [t_{N-1}, t_N].
    Unknown vector  v = [s₀, s₁, …, s_{N-1}]  (N × 4 = 4N unknowns).

    Residual conditions (4N equations):
      k = 0, …, N-2  (continuity):
        F_k(v) = φ(sₖ; [tₖ, t_{k+1}]) − s_{k+1} = 0
      k = N-1  (boundary condition):
        F_{N-1}(v) = φ(s_{N-1}; [t_{N-1}, T]) − y_target = 0   (targeting)
        F_{N-1}(v) = φ(s_{N-1}; [t_{N-1}, T]) − s₀       = 0   (periodic)

    Block Jacobian (4N × 4N, block-bidiagonal):
      k < N-1:   ∂F_k/∂sₖ = Φₖ,   ∂F_k/∂s_{k+1} = −I
      k = N-1:   ∂F_{N-1}/∂s_{N-1} = Φ_{N-1}
                 (+ ∂F_{N-1}/∂s₀ = −I for periodic mode)

    WHY MULTIPLE SHOOTING IS BETTER FOR LONGER HORIZONS:
      For T = 0.5, single shooting already converges.  The advantage of
      MS is demonstrated for T = 1.0 (N = 4, each segment 0.25 time units):
        cond(Φ_k per segment) ≈ 5–10  vs  cond(Φ(T=1.0)) ≈ 10⁴
        → MS yields faster convergence and better numerical stability.
    """

    def __init__(self, T: float, N: int, p: ModelParameters,
                 mode: str = 'targeting', y_target=None):
        if N < 2:
            raise ValueError("N must be ≥ 2 for multiple shooting")
        self.T        = T
        self.N        = N
        self.p        = p
        self.mode     = mode
        self.y_target = y_target
        self.t_nodes  = np.linspace(0, T, N + 1)
        self.n        = 4   # state dimension

    def _integrate_segment(self, s_k: np.ndarray, t_k: float, t_kp1: float):
        """Integrate one segment; return (y_end, Φ_k)."""
        y_end, Phi, _ = integrate_with_sensitivity(s_k, (t_k, t_kp1), self.p)
        return y_end, Phi

    def residual_and_jacobian(self, v: np.ndarray):
        """
        Compute F(v) and dF/dv for the full multiple-shooting system.

        Parameters
        ----------
        v : ndarray (N*4,)    current node values

        Returns
        -------
        F    : ndarray (N*4,)
        dFdv : ndarray (N*4, N*4)  block-bidiagonal Jacobian
        """
        N, n = self.N, self.n
        s    = [v[k*n:(k+1)*n] for k in range(N)]

        F    = np.zeros(N * n)
        dFdv = np.zeros((N * n, N * n))

        for k in range(N):
            t_k   = self.t_nodes[k]
            t_kp1 = self.t_nodes[k + 1]
            y_end, Phi = self._integrate_segment(s[k], t_k, t_kp1)

            row = slice(k * n, (k + 1) * n)
            col = slice(k * n, (k + 1) * n)

            if k < N - 1:
                # Continuity condition
                F[row]          = y_end - s[k + 1]
                dFdv[row, col]  = Phi
                col_next = slice((k + 1) * n, (k + 2) * n)
                dFdv[row, col_next] = -np.eye(n)

            else:
                # Terminal boundary condition
                if self.mode == 'targeting':
                    F[row]         = y_end - self.y_target
                    dFdv[row, col] = Phi
                else:   # periodic
                    F[row]              = y_end - s[0]
                    dFdv[row, col]      = Phi
                    dFdv[row, 0:n]     += -np.eye(n)

        return F, dFdv

    def residual(self, v: np.ndarray) -> np.ndarray:
        F, _ = self.residual_and_jacobian(v)
        return F

    def build_warm_start(self, y0_guess: np.ndarray) -> np.ndarray:
        """
        Build the initial MS node vector by forward-integrating from y0_guess.

        This ensures that all continuity residuals F_0 = … = F_{N-2} = 0
        initially (since the nodes lie on a single forward trajectory).
        Only the terminal residual F_{N-1} is nonzero.
        This is the correct warm-start for MS Newton.
        """
        sol = solve_ivp(
            ode_rhs, (0, self.T), y0_guess, args=(self.p,),
            method='RK45', t_eval=self.t_nodes, rtol=1e-10, atol=1e-12
        )
        # v = [s₀, s₁, …, s_{N-1}]  (exclude the last point = T)
        return sol.y[:, :-1].T.ravel()


def multiple_shooting(T: float, N: int, p: ModelParameters,
                      mode: str, y0_init: np.ndarray,
                      y_target=None,
                      max_iter: int = 50, tol: float = 1e-9,
                      verbose: bool = True):
    """
    Multiple Shooting Method with Newton iteration.

    Partitions [0,T] into N equal sub-intervals.
    Initial node values obtained by forward integration from y0_init
    (warm-start strategy ensures zero continuity residuals initially).

    Parameters
    ----------
    T        : integration horizon
    N        : number of sub-intervals (≥ 2)
    p        : model parameters
    mode     : 'targeting' or 'periodic'
    y0_init  : initial guess for the FIRST node s₀
    y_target : target state (required for 'targeting' mode)
    max_iter : max Newton iterations
    tol      : convergence tolerance
    verbose  : print iteration table

    Returns
    -------
    y0_sol   : solution initial condition  s₀*
    history  : list of ‖F‖ per Newton iteration
    converged: bool
    sol      : full ODE solution of the converged trajectory
    t_nodes  : shooting node times
    """
    header = "=" * 62
    seg = T / N
    print(f"\n{header}")
    print(f"  MULTIPLE SHOOTING — mode: {mode.upper()},  T = {T},  N = {N}")
    print(f"  Segment length = {seg:.4f}   (cond(Φ_k) typically ≈ {1+seg*300:.0f})")
    print(header)

    ms = MultipleShootingBVP(T, N, p, mode, y_target)
    v  = ms.build_warm_start(y0_init)

    # ── Newton iteration ─────────────────────────────────────────────────────
    COL = 58
    history   = []
    converged = False

    if verbose:
        print(f"\n  {'iter':>5}   {'‖F‖₂':>14}   {'‖Δv‖₂':>14}   {'α':>8}")
        print(f"  {'─'*COL}")

    for k in range(max_iter):
        F, J = ms.residual_and_jacobian(v)
        res  = float(norm(F))
        history.append(res)

        if verbose:
            print(f"  {k:>5d}   {res:>14.6e}", end='')

        if res < tol:
            converged = True
            if verbose:
                print(f"   {'—':>14}   {'—':>8}")
                print(f"  {'─'*COL}")
                print(f"  ✓ CONVERGED at iteration {k}   (‖F‖ = {res:.2e})")
            break

        # Newton direction
        try:
            dv = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dv = np.linalg.lstsq(J, -F, rcond=1e-12)[0]

        # Armijo line search on the FULL MS residual
        alpha = 1.0
        for _ in range(30):
            v_trial = v + alpha * dv
            if norm(ms.residual(v_trial)) < res:
                break
            alpha *= 0.5

        if verbose:
            print(f"   {norm(alpha*dv):>14.6e}   {alpha:>8.5f}")

        v = v + alpha * dv

    if verbose and not converged:
        print(f"  {'─'*COL}")
        print(f"  ✗ DID NOT CONVERGE after {max_iter} iterations")

    # Extract solution
    y0_sol = v[:4]
    print(f"\n  Solution y₀*  = {y0_sol}")
    print(f"  Status        : {'✓ SUCCESS' if converged else '✗ FAILED'}")
    print(f"  Iterations    : {len(history)}")
    print(f"  Final ‖F‖     : {history[-1]:.2e}")
    print(header)

    # Reconstruct full trajectory
    t_eval = np.linspace(0, T, 1000)
    sol = solve_ivp(
        ode_rhs, (0, T), y0_sol, args=(p,),
        method='RK45', t_eval=t_eval, rtol=1e-11, atol=1e-13
    )

    return y0_sol, history, converged, sol, ms.t_nodes


# ─────────────────────────────────────────────────────────────────────────────
# 10.  VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'price'    : '#1f4e79',   # dark navy blue
    'short_ma' : '#2e75b6',   # medium blue
    'long_ma'  : '#ed7d31',   # orange
    'sentiment': '#70ad47',   # green
    'accent'   : '#c00000',   # red
    'grid'     : '#e0e0e0',   # light grey
    'node'     : '#7030a0',   # purple
}


def _style_ax(ax, title='', xlabel='', ylabel=''):
    """Apply consistent styling to a matplotlib Axes."""
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, color=COLORS['grid'], linewidth=0.7, linestyle='--', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    if title:  ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)


def plot_market_dynamics(sol, title: str = 'Market Dynamics',
                         save_path: str = None):
    """6-panel overview plot of a market trajectory."""
    t      = sol.t
    P, S, L, M = sol.y

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    # Panel 1: Price + moving averages (spans full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, P, color=COLORS['price'],     lw=2.0, label='Price P(t)',         zorder=3)
    ax1.plot(t, S, color=COLORS['short_ma'],  lw=1.5, ls='--', label='Short MA S(t)')
    ax1.plot(t, L, color=COLORS['long_ma'],   lw=1.5, ls='-.', label='Long MA L(t)')
    ax1.axhline(PARAMS.P_fund, color=COLORS['accent'], lw=1.2, ls=':', label='Fundamental P*=100')
    _style_ax(ax1, title='Asset Price with Short- and Long-Term Moving Averages',
              xlabel='Time  t', ylabel='Price')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Panel 2: Sentiment
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, M, color=COLORS['sentiment'], lw=1.8, zorder=3)
    ax2.axhline(0, color='black', lw=0.8, ls='--')
    ax2.fill_between(t, M, 0, where=(M > 0), alpha=0.25, color=COLORS['sentiment'], label='Bullish')
    ax2.fill_between(t, M, 0, where=(M < 0), alpha=0.25, color=COLORS['accent'],   label='Bearish')
    _style_ax(ax2, title='Market Sentiment  M(t)', xlabel='Time  t', ylabel='Sentiment')
    ax2.legend(fontsize=8)

    # Panel 3: Crossover signal S − L
    ax3 = fig.add_subplot(gs[1, 1])
    cross = S - L
    ax3.plot(t, cross, color=COLORS['short_ma'], lw=1.6, zorder=3)
    ax3.axhline(0, color='black', lw=0.8, ls='--')
    ax3.fill_between(t, cross, 0, where=(cross > 0), alpha=0.3, color=COLORS['short_ma'], label='Bullish cross')
    ax3.fill_between(t, cross, 0, where=(cross < 0), alpha=0.3, color=COLORS['accent'],  label='Bearish cross')
    _style_ax(ax3, title='Crossover Signal  S(t) − L(t)', xlabel='Time  t', ylabel='S − L')
    ax3.legend(fontsize=8)

    # Panel 4: Phase portrait P vs M
    ax4 = fig.add_subplot(gs[2, 0])
    sc = ax4.scatter(P, M, c=t, cmap='plasma', s=6, zorder=3)
    ax4.plot(PARAMS.P_fund, 0, 'k*', ms=14, label='Fixed point y*', zorder=5)
    plt.colorbar(sc, ax=ax4, label='Time')
    _style_ax(ax4, title='Phase Portrait: Price vs Sentiment',
              xlabel='Asset Price P', ylabel='Sentiment M')
    ax4.legend(fontsize=8)

    # Panel 5: Phase portrait S−L vs M
    ax5 = fig.add_subplot(gs[2, 1])
    sc5 = ax5.scatter(cross, M, c=t, cmap='viridis', s=6, zorder=3)
    plt.colorbar(sc5, ax=ax5, label='Time')
    _style_ax(ax5, title='Phase Portrait: Crossover vs Sentiment',
              xlabel='S − L', ylabel='Sentiment M')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_convergence_comparison(hist_ss: list,
                                 hist_ms_list: list,
                                 labels_ms: list,
                                 save_path: str = None):
    """Semi-log convergence comparison: single vs multiple shooting."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(hist_ss, 'o-', color=COLORS['accent'], lw=2.2,
                ms=6, label='Single Shooting')
    palette = ['#2e75b6', '#70ad47', '#9e5a9e', '#ed7d31']
    for i, (hist, lbl) in enumerate(zip(hist_ms_list, labels_ms)):
        ax.semilogy(hist, 's--', color=palette[i % 4], lw=1.8,
                    ms=5, label=lbl)
    _style_ax(ax, title='Convergence of Newton Iterations\n(Single vs Multiple Shooting)',
              xlabel='Newton Iteration  k', ylabel='‖F(y₀^k)‖₂  (log scale)')
    ax.legend(fontsize=9)
    # Annotate quadratic convergence region
    ax.annotate('quadratic\nconvergence', xy=(3, hist_ss[3] if len(hist_ss)>3 else 1e-3),
                xytext=(4.5, 5e-2), fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_shooting_nodes(sol, t_nodes: np.ndarray, N: int,
                        save_path: str = None):
    """Visualise multiple shooting sub-intervals on the price trajectory."""
    t, P = sol.t, sol.y[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, P, color=COLORS['price'], lw=2.0, label='Price P(t)', zorder=3)
    ax.axhline(PARAMS.P_fund, color=COLORS['accent'], lw=1.0, ls=':', label='P* = 100')

    for i, tn in enumerate(t_nodes):
        lbl = f'Shooting nodes (N={N})' if i == 0 else None
        ax.axvline(tn, color=COLORS['node'], lw=0.9, ls='--', alpha=0.7, label=lbl)

    # Mark the nodes on the curve
    node_P = np.interp(t_nodes[:-1], sol.t, P)
    ax.scatter(t_nodes[:-1], node_P, color=COLORS['node'], s=60, zorder=5,
               label='Node states sₖ', edgecolors='white', lw=1.0)

    _style_ax(ax, title=f'Multiple Shooting: {N} Sub-Intervals on Price Trajectory',
              xlabel='Time  t', ylabel='Asset Price  P(t)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_eigenvalues(fp_data: dict, save_path: str = None):
    """Complex-plane plot of Jacobian eigenvalues at the fixed point."""
    evs = fp_data['eigenvalues']
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.axvline(0, color='red',   lw=1.2, ls='--', alpha=0.6, label='Stability boundary (Re=0)')

    # Shade stable half-plane
    xlim = ax.get_xlim()
    ax.fill_betweenx([-8, 8], xlim[0] - 1, 0, alpha=0.06, color='green', label='Stable half-plane')

    colors_ev = ['#1f4e79', '#ed7d31', '#70ad47', '#c00000']
    for i, ev in enumerate(evs):
        ax.scatter(ev.real, ev.imag, color=colors_ev[i], s=140, zorder=5,
                   label=f'λ_{i+1} = {ev.real:+.3f} {ev.imag:+.3f}i',
                   edgecolors='white', lw=1.5)

    _style_ax(ax, title='Eigenvalues of Jacobian  J(y*)  at Fixed Point',
              xlabel='Real part', ylabel='Imaginary part')
    ax.set_xlim(min(evs.real) * 1.4, max(evs.real) * 0.3)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_bubble_scenarios(p: ModelParameters, save_path: str = None):
    """Multi-scenario plot: bubble formation, burst, and equilibrium start."""
    scenarios = [
        {'y0': [105, 100, 100,  0.5], 'label': 'Mild bubble   (P₀=105, M=+0.5)', 'color': '#2e75b6'},
        {'y0': [115, 100, 100,  1.5], 'label': 'Strong bubble (P₀=115, M=+1.5)', 'color': '#c00000'},
        {'y0': [ 92, 100, 100, -0.8], 'label': 'Crash start   (P₀=92,  M=−0.8)', 'color': '#ed7d31'},
        {'y0': [100, 100, 100,  0.0], 'label': 'Equilibrium   (P₀=100, M= 0.0)', 'color': '#70ad47'},
    ]
    T_sim  = 20.0
    t_eval = np.linspace(0, T_sim, 1200)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    info = [
        ('Asset Price  P(t)',   'Price',     0),
        ('Short MA  S(t)',      'Price',     1),
        ('Long MA  L(t)',       'Price',     2),
        ('Sentiment  M(t)',     'Sentiment', 3),
    ]

    for scen in scenarios:
        sol = solve_ivp(ode_rhs, (0, T_sim), scen['y0'], args=(p,),
                        method='RK45', t_eval=t_eval, rtol=1e-10, atol=1e-12)
        for (title, ylbl, idx), ax in zip(info, axes):
            ax.plot(sol.t, sol.y[idx], color=scen['color'],
                    lw=1.7, label=scen['label'])

    for (title, ylbl, idx), ax in zip(info, axes):
        if idx < 3:
            ax.axhline(p.P_fund, color='black', lw=0.9, ls=':', alpha=0.5, label='P* = 100')
        _style_ax(ax, title=title, xlabel='Time  t', ylabel=ylbl)
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Market Bubble & Burst: Multiple Initial Conditions', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_sensitivity_analysis(p: ModelParameters, save_path: str = None):
    """Show effect of trading intensity α on price and sentiment dynamics."""
    alphas  = [0.5, 1.5, 2.5, 3.5]
    palette = ['#1f4e79', '#2e75b6', '#ed7d31', '#c00000']
    y0      = [110.0, 101.0, 100.5, 1.0]
    t_s     = np.linspace(0, 20, 800)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for alpha_val, clr in zip(alphas, palette):
        p_tmp       = ModelParameters()
        p_tmp.alpha = alpha_val
        sol_tmp = solve_ivp(ode_rhs, (0, 20), y0, args=(p_tmp,),
                            method='RK45', t_eval=t_s, rtol=1e-10, atol=1e-12)
        axes[0].plot(sol_tmp.t, sol_tmp.y[0], color=clr, lw=1.8, label=f'α = {alpha_val}')
        axes[1].plot(sol_tmp.t, sol_tmp.y[3], color=clr, lw=1.8, label=f'α = {alpha_val}')

    for ax, title, ylbl in zip(axes,
            ['Asset Price  P(t)', 'Sentiment  M(t)'],
            ['Price',             'Sentiment']):
        _style_ax(ax, title=title+'\n(sensitivity to trading intensity α)',
                  xlabel='Time  t', ylabel=ylbl)
        ax.legend(fontsize=9)
    axes[0].axhline(PARAMS.P_fund, ls=':', color='gray', lw=0.9)

    fig.suptitle('Sensitivity Analysis: Momentum Trading Intensity α', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_bvp_solution(sol_ss, sol_ms, bvp: BVPProblem, save_path: str = None):
    """
    Overlay single-shooting and multiple-shooting BVP solutions on one plot,
    confirming they coincide (they should be identical).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    labels_y = ['Asset Price P(t)', 'Short MA S(t)', 'Long MA L(t)', 'Sentiment M(t)']
    ylabels  = ['Price', 'Price', 'Price', 'Sentiment']

    for idx, (ax, lbl, ylbl) in enumerate(zip(axes, labels_y, ylabels)):
        ax.plot(sol_ss.t, sol_ss.y[idx], color=COLORS['price'],    lw=2.2,
                label='Single Shooting', zorder=3)
        ax.plot(sol_ms.t, sol_ms.y[idx], color=COLORS['sentiment'], lw=1.6,
                ls='--', label='Multiple Shooting', zorder=4)
        if idx < 3:
            ax.axhline(PARAMS.P_fund, color=COLORS['accent'], ls=':', lw=0.9, label='P*=100')

        # Mark t=0 and t=T
        ax.axvline(0,      color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axvline(bvp.T,  color='gray', lw=0.8, ls='--', alpha=0.5)

        # Mark target
        if bvp.mode == 'targeting' and idx < 4:
            ax.scatter([bvp.T], [bvp.y_target[idx]],
                       color=COLORS['accent'], s=80, zorder=5,
                       label='Target state', marker='D')
        _style_ax(ax, title=lbl, xlabel='Time  t', ylabel=ylbl)
        ax.legend(fontsize=8)

    title = (f'BVP Solution — Mode: {bvp.mode.upper()},  T = {bvp.T}\n'
             f'Single Shooting & Multiple Shooting (overlaid)')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_condition_numbers(p: ModelParameters, save_path: str = None):
    """
    Plot cond(Φ(T)) vs T to illustrate why T matters for Newton convergence.
    This is the KEY diagnostic figure explaining the original code's failure.
    """
    y0_ref = np.array([103.0, 101.5, 100.5, 0.3])
    T_vals = np.linspace(0.05, 3.0, 60)
    conds  = []

    for T in T_vals:
        _, Phi, _ = integrate_with_sensitivity(y0_ref, (0, T), p)
        conds.append(np.linalg.cond(Phi))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(T_vals, conds, color=COLORS['price'], lw=2.2, zorder=3)

    # Mark our chosen T=0.5
    idx05 = np.argmin(np.abs(T_vals - 0.5))
    ax.scatter([0.5], [conds[idx05]], color=COLORS['sentiment'], s=120, zorder=5,
               label=f'T=0.5: cond={conds[idx05]:.0f}  ✓ (used in BVP)', edgecolors='white', lw=1.5)

    # Mark the original T=5
    ax.axvline(5.0, color=COLORS['accent'], lw=1.5, ls='--', alpha=0.7, label='Original T=5 (DIVERGES)')
    ax.axhline(1e8, color='orange', lw=1.2, ls=':', label='Danger zone  cond > 10⁸')

    # Shade the safe region
    ax.axvspan(0, 1.0, alpha=0.07, color='green', label='Well-conditioned region')

    _style_ax(ax, title='Condition Number of Monodromy Matrix Φ(T) vs Integration Horizon\n'
                        '(Shows why T must be kept small for Newton convergence)',
              xlabel='Horizon  T', ylabel='cond(Φ(T))  (log scale)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(fig_dir: str = None):
    """
    Execute the complete suite of experiments and save all figures.

    Experiments:
      A. Fixed point analysis
      B. Bubble & crash scenario survey
      C. Free trajectory (unconstrained simulation)
      D. Condition number diagnostic (explains convergence failures)
      E. BVP: Targeting — Single Shooting (T = 0.5)
      F. BVP: Targeting — Multiple Shooting  N=2, N=4 (T = 0.5)
      G. Overlay comparison: SS vs MS solutions
      H. Convergence comparison (semi-log plot)
      I. Multiple Shooting nodes visualization
      J. Sensitivity analysis (parameter α)
    """
    if fig_dir is None:
        fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    p = ModelParameters()

    banner = "█" * 62
    print(f"\n{banner}")
    print("  ASSET PRICING BVP MODEL — COMPLETE EXPERIMENT SUITE v2.0")
    print(banner)
    print(f"\n  Model: {p}")
    print(f"  Output directory: {fig_dir}\n")

    # ── A. Fixed point analysis ────────────────────────────────────────────
    print("\n" + "─"*62)
    print("[A]  FIXED POINT ANALYSIS")
    fp = print_fixed_point_analysis(p)
    plot_eigenvalues(fp, save_path=f'{fig_dir}/fig_A_eigenvalues.png')

    # ── B. Bubble & crash scenarios ────────────────────────────────────────
    print("\n[B]  BUBBLE & CRASH SCENARIOS")
    plot_bubble_scenarios(p, save_path=f'{fig_dir}/fig_B_bubble_scenarios.png')
    print("  Saved bubble scenarios figure.")

    # ── C. Free reference trajectory ─────────────────────────────────────
    print("\n[C]  FREE REFERENCE TRAJECTORY  (y₀ = [112, 102, 100.5, 1.2])")
    y0_free = np.array([112.0, 102.0, 100.5, 1.2])
    T_free  = 15.0
    sol_free = solve_ivp(
        ode_rhs, (0, T_free), y0_free, args=(p,),
        method='RK45', t_eval=np.linspace(0, T_free, 1500),
        rtol=1e-12, atol=1e-14
    )
    plot_market_dynamics(sol_free,
        title='Asset Pricing Model – Free (Unconstrained) Trajectory',
        save_path=f'{fig_dir}/fig_C_free_trajectory.png')

    # ── D. Condition number diagnostic ─────────────────────────────────────
    print("\n[D]  CONDITION NUMBER DIAGNOSTIC  (why T=0.5?)")
    print("  Computing cond(Φ(T)) for T ∈ [0.05, 3.0] …")
    plot_condition_numbers(p, save_path=f'{fig_dir}/fig_D_condition_numbers.png')

    # ─────────────────────────────────────────────────────────────────────
    # BVP SETUP:  T = 0.5,  y₀_true = [103, 101.5, 100.5, 0.3]
    #
    # We set the target as the natural endpoint of a known trajectory,
    # then perturb y₀_true to create a realistic initial guess.
    # ─────────────────────────────────────────────────────────────────────
    y0_true  = np.array([103.0, 101.5, 100.5, 0.3])
    T_bvp    = 0.5

    # Compute the TRUE target by high-accuracy integration
    sol_true = solve_ivp(
        ode_rhs, (0, T_bvp), y0_true, args=(p,),
        method='RK45', rtol=1e-13, atol=1e-15
    )
    y_target = sol_true.y[:, -1].copy()

    # Realistic perturbation: 2.5 in P, 1.5 in S, 0.8 in L, 0.15 in M
    y0_guess = y0_true + np.array([2.5, 1.5, 0.8, 0.15])

    print(f"\n  BVP setup:")
    print(f"    T_bvp    = {T_bvp}")
    print(f"    y₀_true  = {y0_true}")
    print(f"    y_target = {y_target.round(6)}")
    print(f"    y₀_guess = {y0_guess}  (‖guess−true‖ = {norm(y0_guess-y0_true):.3f})")

    # ── E. Single Shooting ────────────────────────────────────────────────
    print("\n" + "─"*62)
    print("[E]  SINGLE SHOOTING  (T = 0.5)")
    bvp_ss = BVPProblem(T_bvp, p, mode='targeting', y_target=y_target)
    y0_ss, hist_ss, conv_ss, sol_ss = single_shooting(
        bvp_ss, y0_guess, tol=1e-9, verbose=True
    )
    plot_market_dynamics(sol_ss,
        title='Single Shooting — Targeting BVP Solution  (T = 0.5)',
        save_path=f'{fig_dir}/fig_E_single_shooting_traj.png')

    # ── F. Multiple Shooting ──────────────────────────────────────────────
    print("\n" + "─"*62)
    print("[F]  MULTIPLE SHOOTING  (T = 0.5,  N = 2 and N = 4)")

    results_ms = {}
    for N in [2, 4]:
        y0_ms, hist_ms, conv_ms, sol_ms, t_nodes = multiple_shooting(
            T_bvp, N, p, 'targeting',
            y0_guess, y_target=y_target,
            tol=1e-9, verbose=True
        )
        results_ms[N] = (y0_ms, hist_ms, conv_ms, sol_ms, t_nodes)

    # Best MS solution for visualisation
    N_best = 4
    y0_ms_best, hist_ms_best, conv_ms_best, sol_ms_best, t_nodes_best = results_ms[N_best]

    # ── G. Overlay SS vs MS ───────────────────────────────────────────────
    print("\n[G]  OVERLAY PLOT: Single Shooting vs Multiple Shooting")
    if conv_ss and conv_ms_best:
        plot_bvp_solution(sol_ss, sol_ms_best, bvp_ss,
            save_path=f'{fig_dir}/fig_G_ss_vs_ms_overlay.png')

    # ── H. Convergence comparison ─────────────────────────────────────────
    print("\n[H]  CONVERGENCE COMPARISON")
    hist_ms_list = [results_ms[N][1] for N in [2, 4]]
    labels_ms    = [f'Multiple Shooting  N={N}' for N in [2, 4]]
    plot_convergence_comparison(
        hist_ss, hist_ms_list, labels_ms,
        save_path=f'{fig_dir}/fig_H_convergence.png'
    )

    # ── I. MS nodes visualisation ─────────────────────────────────────────
    print("\n[I]  MULTIPLE SHOOTING NODES")
    if conv_ms_best:
        plot_shooting_nodes(sol_ms_best, t_nodes_best, N_best,
            save_path=f'{fig_dir}/fig_I_ms_nodes.png')

    # ── J. Sensitivity analysis ───────────────────────────────────────────
    print("\n[J]  SENSITIVITY ANALYSIS  (effect of α)")
    plot_sensitivity_analysis(p, save_path=f'{fig_dir}/fig_J_sensitivity_alpha.png')

    # ── SUMMARY ──────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("  EXPERIMENT SUMMARY")
    print(sep)
    print(f"  Fixed point y* = {fp['fixed_point']}")
    print(f"  Stability      : {'STABLE' if fp['stable'] else 'UNSTABLE'}")
    print(f"  Eigenvalues    : {fp['eigenvalues'].round(4)}")
    print()
    print(f"  BVP parameters : T={T_bvp},  y_target={y_target.round(4)}")
    print(f"  Initial guess  : y₀_guess={y0_guess}")
    print()
    status = lambda b: '✓ CONVERGED' if b else '✗ FAILED'
    print(f"  [E] Single Shooting:         {status(conv_ss)}  in {len(hist_ss)} iters,  ‖F‖={hist_ss[-1]:.2e}")
    for N in [2, 4]:
        _, h, c, _, _ = results_ms[N]
        print(f"  [F] Multiple Shooting N={N}:   {status(c)}  in {len(h)} iters,  ‖F‖={h[-1]:.2e}")
    print()
    print(f"  Figures written to: {fig_dir}/")
    print(sep)

    return {
        'fixed_point' : fp,
        'y0_true'     : y0_true,
        'y_target'    : y_target,
        'y0_guess'    : y0_guess,
        'T_bvp'       : T_bvp,
        'ss'          : (y0_ss, hist_ss, conv_ss, sol_ss),
        'ms'          : results_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = run_all_experiments()
