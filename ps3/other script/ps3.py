
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


formula = (
    "ln_sj_s0 ~  x + ln_within_sat + ln_within_wired "
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

