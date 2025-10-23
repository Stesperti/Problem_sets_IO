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
        # Guard: own-prices derivatives should be negative
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
        nest_ids = 1 if (j + 1 ) == 1 or (j + 1) == 0 else 1
        records.append({
            "market_ids": t+1,
            "product_ids": j+1,
            "nesting_ids": nest_ids,
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
        "prices":    prices_root[:, j],
        "x":        x[:, j],
        "w":        w[:, j],
    })
    # y = β0 + βx·x + βp·price, instrument prices with w
    res = IV2SLS.from_formula("y ~ 1 + x + [prices ~ w]", data=df_j).fit(cov_type="robust")
    print(f"2SLS results, product {j+1}")
    print(res.summary) 

#%%
df = pd.read_csv("paytv_sim_equilibrium.csv")

# df = pd.DataFrame.from_records(records)
eps = 1e-12

# 1) Outside share and log share ratio
mkt_sum = df.groupby("market_ids")["shares"].sum().rename("sum_share_mkt")
df = df.merge(mkt_sum, on="market_ids", how="left")
df["s0"] = (1.0 - df["sum_share_mkt"]).clip(lower=eps)
df["ln_sj_s0"] = np.log(df["shares"].clip(lower=eps)) - np.log(df["s0"])

# 2) Boolean masks for nests
mask_sat = df["is_satellite"].astype(bool)
mask_wir = df["is_wired"].astype(bool)

# 3) Nest totals S_{g,t}
S_sat = (df.assign(sat_share=np.where(mask_sat, df["shares"], 0.0))
           .groupby("market_ids")["sat_share"].transform("sum"))
S_wir = (df.assign(wir_share=np.where(mask_wir, df["shares"], 0.0))
           .groupby("market_ids")["wir_share"].transform("sum"))

# 4) Within-nest logs (0 for products not in that nest)
df["ln_within_sat"] = 0.0
valid_sat = mask_sat & (S_sat > 0)
df.loc[valid_sat, "ln_within_sat"] = np.log((df.loc[valid_sat, "shares"] / S_sat[valid_sat]).clip(lower=eps))

df["ln_within_wired"] = 0.0
valid_wir = mask_wir & (S_wir > 0)
df.loc[valid_wir, "ln_within_wired"] = np.log((df.loc[valid_wir, "shares"] / S_wir[valid_wir]).clip(lower=eps))

# 5) 2SLS: instrument price only (keep your names)
# Model: ln(s_j/s_0) = β*x + α*prices + σ_sat*ln(s|sat) + σ_wir*ln(s|wir) + ξ
# Endog: prices; Exog: 1 + x + ln_within_sat + ln_within_wired; IVs: exog + w
formula = (
    "ln_sj_s0 ~ 1 + x + ln_within_sat + ln_within_wired "
    "[prices ~ w]"
)
iv_res = IV2SLS.from_formula(formula, data=df).fit(
    cov_type="clustered", clusters=df["market_ids"]
)
print(iv_res.summary)

# Coefs
beta_x    = iv_res.params["x"]
alpha_p   = iv_res.params["prices"]
sigma_sat = iv_res.params["ln_within_sat"]
sigma_wir = iv_res.params["ln_within_wired"]
print("\nEstimates:")
print(f"Beta on x           = {beta_x:.4f}")
print(f"Alpha on prices     = {alpha_p:.4f}")
print(f"Sigma (satellite)   = {sigma_sat:.4f}")
print(f"Sigma (wired)       = {sigma_wir:.4f}")

# %% 7 Construct a latex table iwth the estimate


# %% 8
df = pd.read_csv("paytv_sim_equilibrium.csv")
df.head()
RNG_SEED = 2025
np.random.seed(RNG_SEED)

T = 600                 # markets
J = 4                   # products per market
N = 20               # number of simulation draws for agents (Monte Carlo)

# True parameters
beta1 = 1.0             # on observed quality x
beta_sat_mean = 4.0     # mean taste for satellite indicator
beta_wir_mean = 4.0     # mean taste for wired indicator
alpha = -2.0            # price coefficient (fixed)

# Random-coeff std (satellite, wired)
sigma_rc = np.array([1.0, 1.0])  # diag entries

# Supply side: log mc = gamma0 + gamma1 * w + omega/8
gamma0 = 0.5
gamma1 = 0.25

# Correlation between xi and omega is 0.25, variances are 1
cov_xi_omega = 0.25
cov = np.array([[1.0, cov_xi_omega],
                [cov_xi_omega, 1.0]])

# ----------------------------
# 1) Build product-level panel
# ----------------------------
markets = np.repeat(np.arange(T), J)
products = np.tile(np.arange(1, J+1), T)

# Indicators: products 1-2 are satellite, 3-4 are wired
is_sat = (products <= 2).astype(float)
is_wir = (products >= 3).astype(float)

# Observed demand shifter x_jt and cost shifter w_jt: abs(N(0,1))
x = np.abs(np.random.normal(size=T*J))
w = np.abs(np.random.normal(size=T*J))

# Demand and cost unobservables (xi, omega) with correlation 0.25
xi_omega = np.random.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=T*J)
xi = xi_omega[:, 0]
omega = xi_omega[:, 1]

# Log marginal costs and MC level
log_mc = gamma0 + gamma1 * w + omega / 8.0
mc = np.exp(log_mc)

# ----------------------------
# 2) Set up pyBLP structures
# ----------------------------
# Demand: X1 has [x, satellite, wired]; price is in 'prices' column (handled separately)
# Random coefficients on [satellite, wired]
# Supply: X3 has [1, w] (linear-in-parameters log cost)
form_X1 = pyblp.Formulation('0 + x + satellite + wired')
form_X2 = pyblp.Formulation('0 + satellite + wired')  # RC on these
form_X3 = pyblp.Formulation('1 + w')                  # supply shifters (log cost)

# Product identifiers
firm_ids = products  # single-product firms: each product is its own firm

# Assemble product DataFrame required by pyBLP
prod = pd.DataFrame({
    'market_ids': markets,
    'product_ids': products + markets * 10,  # unique id per product-market
    'firm_ids': firm_ids,
    'x': x,
    'satellite': is_sat,
    'wired': is_wir,
    'w': w
})

