# %%
import numpy as np
import pyblp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.optimize import root
pyblp.options.digits = 2
pyblp.options.verbose = False
pyblp.__version__

# %%
def foc_norm(p, x_t, xi_t, mc_t):
    r = foc_residuals(p, x_t, xi_t, mc_t)
    return np.max(np.abs(r)), r

def ms_mapping_gap(p, x_t, xi_t, mc_t):
    s, dsdp = sim_shares_and_jacobian(p, x_t, xi_t)
    gap = p - (mc_t - s / np.diag(dsdp))  # zero at fixed point
    return np.max(np.abs(gap)), gap

def sim_shares_and_jacobian(p_t, x_t, xi_t):
    """
    Inputs:
    p_t  : (J,) prices in market t
    x_t  : (J,) quality draws for market t
    xi_t : (J,) demand unobservable for market t
    Returns:
    s_mean : (J,) simulated market shares (inside goods)
    dsdp   : (J, J) Jacobian ds_j/dp_k
    Details:
    Individual utility (up to i.i.d. EV1): v_ij = beta1*x_j + beta2_i*sat_j + beta3_i*wir_j + alpha*p_j + xi_j
    s_ij = exp(v_ij) / (1 + sum_k exp(v_ik))  (outside utility normalized to 0)
    Aggregate s_j = E_i[s_ij] via R-draw simulation.
    Derivative under the integral (per draw): ∂s_ij/∂p_k = alpha * s_ij * (1{j=k} - s_ik)
    => dsdp[j,k] = E_i[ ∂s_ij/∂p_k ] (Monte Carlo average).
    """
    # v_i (R,J): random coeffs enter via sat/wir dummies
    v = (beta1 * x_t[None, :]
        + beta2_draws[:, None] * is_sat[None, :]
        + beta3_draws[:, None] * is_wir[None, :]
        + alpha * p_t[None, :]
        + xi_t[None, :])  # shape (R, J)

    exp_v = np.exp(v - np.max(v, axis=1, keepdims=True))  # safe stab.
    denom = 1.0 + np.sum(exp_v, axis=1, keepdims=True)    # add outside option
    s_ind = exp_v / denom                                 # (R, J)
    s_mean = s_ind.mean(axis=0)                           # (J,)

    # Jacobian: ds_j/dp_k = E_i[ alpha * s_ij * (1{j=k} - s_ik) ]
    dsdp = np.empty((J, J))
    for k in range(J):
        factor = alpha * (np.eye(J)[k][None, :] - s_ind)  
        dsdp[:, k] = np.mean(s_ind * factor, axis=0)
    return s_mean, dsdp

def foc_residuals(p_t, x_t, xi_t, mc_t):
    s, dsdp = sim_shares_and_jacobian(p_t, x_t, xi_t)
    # Single-product FOCs: (p_j - mc_j)*ds_j/dp_j + s_j = 0  for each j
    diag = np.diag(dsdp)  
    return (p_t - mc_t) * diag + s

def solve_equilibrium_root(x_t, xi_t, mc_t, p0=None, tol=1e-12):
    # Root-finding on the J FOCs
    if p0 is None:
        p0 = mc_t + 1.0  # mild markup starting point
    sol = root(lambda p: foc_residuals(p, x_t, xi_t, mc_t), p0, method='hybr', tol=tol)
    return sol.x, sol.success, sol.nfev

# ------------------------------------------------------
# Morrow–Skerlos Fixed-Point (MSFP) mapping per market
# ------------------------------------------------------
def msfp_prices(x_t, xi_t, mc_t, p0=None, max_iter=5000, tol=1e-10, damp=0.8):
    """
    p_{n+1} = mc - diag(dsdp(p_n))^{-1} s(p_n)
    Optional damping for stability.
    """
    if p0 is None:
        p = mc_t + 1.0
    else:
        p = p0.copy()
    for it in range(max_iter):
        s, dsdp = sim_shares_and_jacobian(p, x_t, xi_t)
        diag = np.diag(dsdp)
        # Guard: own-price derivatives should be negative
        if np.any(diag >= 0):
            # If happens, try to bail with small step toward mc
            p = 0.5 * (p + mc_t)
            continue
        p_new = mc_t - s / diag
        # damping
        p_next = damp * p_new + (1.0 - damp) * p
        # convergence
        if np.max(np.abs(p_next - p)) < tol:
            return p_next, True, it + 1
        p = p_next
    return p, False, max_iter

# %%

rng = default_rng(12345)  # reproducible
T = 600                    # markets
J = 4                      
R = 100              
# Demand parameters
beta1 = 1.0                
alpha = -2.0               
# random coefficients: beta2_i ~ N(4,1) on satellite, beta3_i ~ N(4,1) on wired
mu_sat, sd_sat = 4.0, 1.0
mu_wir, sd_wir = 4.0, 1.0
# Cost parameters
gamma0 = 0.5
gamma1 = 0.25
# Correlated unobservables (xi, omega): N(0,0; 1, 0.25; 0.25, 1)
Sigma = np.array([[1.0, 0.25],
                [0.25, 1.0]])
# Product identities: j=0,1 are satellite; j=2,3 are wired
is_sat = np.array([1, 1, 0, 0], dtype=int)
is_wir = 1 - is_sat

# Exogenous characteristics and cost shifter: abs(N(0,1))
x = np.abs(rng.standard_normal((T, J)))
w = np.abs(rng.standard_normal((T, J)))

z = rng.multivariate_normal(mean=np.zeros(2), cov=Sigma, size=T*J)
xi = z[:, 0].reshape(T, J)
omega = z[:, 1].reshape(T, J)

# Marginal costs: ln mc_jt = gamma0 + gamma1 * w_jt + omega_jt/8
mc = np.exp(gamma0 + gamma1 * w + omega / 8.0)  # shape (T, J)

beta2_draws = mu_sat + sd_sat * rng.standard_normal(R)   # satellite taste
beta3_draws = mu_wir + sd_wir * rng.standard_normal(R)   # wired taste



# -------------------------
prices_root = np.empty((T, J))
succ_root   = np.zeros(T, dtype=bool)
evals_root  = np.zeros(T, dtype=int)

prices_msfp = np.empty((T, J))
succ_msfp   = np.zeros(T, dtype=bool)
iters_msfp  = np.zeros(T, dtype=int)

for t in range(T):
    # Root-solver

    # MSFP, warm-start at root solution (or mc+1 if root failed)
    p0 = (mc[t] + 1.0)
    p_ms, ok_ms, it_ms = msfp_prices(x[t], xi[t], mc[t], p0=p0, max_iter=2000, tol=1e-12, damp=0.85)
    prices_msfp[t] = p_ms
    succ_msfp[t] = ok_ms
    iters_msfp[t] = it_ms

    p_star, ok, nfev = solve_equilibrium_root(x[t], xi[t], mc[t], p0=p_ms)
    prices_root[t] = p_star
    succ_root[t] = ok
    evals_root[t] = nfev


# -------------------------------------------
shares_root = np.empty((T, J))
shares_msfp = np.empty((T, J))
for t in range(T):
    s_r, _ = sim_shares_and_jacobian(prices_root[t], x[t], xi[t])
    s_m, _ = sim_shares_and_jacobian(prices_msfp[t], x[t], xi[t])
    shares_root[t] = s_r
    shares_msfp[t] = s_m

# -------------------------
print(f"Root-solver success: {succ_root.mean():.3f} of markets")
print(f"MSFP success:        {succ_msfp.mean():.3f} of markets")

# Compare the two methods (they should match very closely if both converged)
diff = np.abs(prices_root - prices_msfp)
print("Max |price_root - price_msfp|:", np.nanmax(diff))

t = 0  # pick a market to inspect
p_r = prices_root[t]
p_m = prices_msfp[t]

fn_r, rvec_r = foc_norm(p_r, x[t], xi[t], mc[t])
fn_m, rvec_m = foc_norm(p_m, x[t], xi[t], mc[t])
mg_r, gvec_r = ms_mapping_gap(p_r, x[t], xi[t], mc[t])
mg_m, gvec_m = ms_mapping_gap(p_m, x[t], xi[t], mc[t])

print("Root solution:  max FOC residual =", fn_r, " | max MS gap =", mg_r)
print("MSFP solution:  max FOC residual =", fn_m, " | max MS gap =", mg_m)
print("max |p_root - p_msfp| =", np.max(np.abs(p_r - p_m)))

# Optional: assemble a tidy DataFrame to export
records = []
for t in range(T):
    for j in range(J):
        records.append({
            "market_ids": t+1,
            "product_ids": j+1,
            "is_satellite": is_sat[j],
            "is_wired": is_wir[j],
            "x": x[t, j],
            "w": w[t, j],
            "xi": xi[t, j],
            "omega": omega[t, j],
            "mc": mc[t, j],
            "prices": prices_root[t, j],
            "shares": shares_root[t, j],
        })
df = pd.DataFrame.from_records(records)
df.to_csv("paytv_sim_equilibrium.csv", index=False)
print(df.head())


# %%
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
ln_s_0t = - np.log(1.0 - shares_root.sum(axis=1))  # (T,)
diff_ln_s_jt = np.log(shares_root) - np.repeat(ln_s_0t[:, None], J, axis=1)  # (T,J)

#OLS regression of diff_ln_s_jt on x and prices

for j in range(J):
    plt.scatter(prices_root[:, j], diff_ln_s_jt[:, j], alpha=0.5, label=f'Product {j+1}')
    plt.xlabel('Prices')
    plt.ylabel('diff_ln_s_jt')
    plt.title('diff_ln_s_jt vs Prices')
    plt.show()
    X_j = sm.add_constant(np.column_stack((x[:, j], prices_root[:, j])))
    model_j = sm.OLS(diff_ln_s_jt[:, j], X_j)
    results_j = model_j.fit()
    print(f'Regression results for Product {j+1}:')
    print(results_j.summary())  

#estimation using two-stage least squares (2SLS) with w as an instrument for prices
    df_j = pd.DataFrame({
        "y":        diff_ln_s_jt[:, j],
        "price":    prices_root[:, j],
        "x":        x[:, j],
        "w":        w[:, j],
    })
    # y = β0 + βx·x + βp·price, instrument price with w
    res = IV2SLS.from_formula("y ~ 1 + x + [price ~ w]", data=df_j).fit(cov_type="robust")
    print(f"2SLS results, product {j+1}")
    print(res.summary) 

# %%
