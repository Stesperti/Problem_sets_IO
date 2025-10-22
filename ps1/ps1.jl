### Problem Set 1
# Author: Stefano Sperti

using DataFrames, CSV, Plots, Statistics, Random, Distributions, Optim, LinearAlgebra, DataFrames, Latexify, FastGaussQuadrature, ForwardDiff

cd(@__DIR__)
figures_dir = joinpath(@__DIR__, "figures")
if !isdir(figures_dir)
    mkdir(figures_dir)
end
# Create latex directory if it doesn't exist
latex_dir = joinpath(@__DIR__, "latex")
if !isdir(latex_dir)
    mkdir(latex_dir)
end
println("Starting analysis...")


options = Optim.Options(
    g_tol=1e-6,          # gradient tolerance
    iterations=500,     # max iterations
    show_trace=true,
    show_every=50       # print trace every 50 iterations
)

school_dataset = CSV.read("schools_dataset.csv", DataFrame)
println("Number of rows: ", size(school_dataset, 1))
println("Number of columns: ", size(school_dataset, 2))
println("Column names: ", names(school_dataset))
println("First 5 rows: ")

### 1
println("--------------------------------")
println("Printing distribution of distance to all schools, and the distribution of distance to the chosen school.")
println("--------------------------------")

histogram(school_dataset[!, :distance], bins=20, title="Histogram of Distance to all the Schools", xlabel="Distance from schools School", ylabel="Frequency")
savefig(joinpath(figures_dir, "histogram_distance_all_schools.png"))

min_distances_vector = combine(groupby(school_dataset, :household_id),
    :distance => minimum).distance_minimum
histogram(min_distances_vector, bins=20, title="Histogram of Minimum Distance to Schools", xlabel="Minimum Distance from schools School", ylabel="Frequency")
savefig(joinpath(figures_dir, "histogram_min_distance_schools.png"))


counts = combine(groupby(school_dataset, :school_id), nrow => :count)
different_counts = unique(groupby(school_dataset, :school_id))
println("Number of unique counts of observations per school: ", size(different_counts, 1))
println("The data is balanced: ", counts)

###
println("Preparing data for estimation...")
println("Number of unique households: ", length(unique(school_dataset[:, :household_id])))
println("Number of unique schools: ", length(unique(school_dataset[:, :school_id])))

# School characteristics
school_info = unique(school_dataset[:, [:school_id, :test_scores, :sports]])
println("School info dimensions: ", size(school_info))


rename!(school_info, [:test_scores => :test_scores_school, :sports => :sports_school])
school_info = sort(school_info, :school_id)
test_scores = school_info[:, :test_scores_school]
sports = school_info[:, :sports_school]

# Distance matrix (households × schools)
dist_df = unstack(school_dataset, :household_id, :school_id, :distance)
dist_df = sort(dist_df, :household_id)
distance = Matrix(dist_df[:, Not(:household_id)])

# Choice vector y: chosen school per household
household_ids = sort(unique(school_dataset[:, :household_id]))
y = [school_dataset[(school_dataset.household_id.==h).&(school_dataset.y_ij.==1), :school_id][1] for h in household_ids]

# If school_id is not 1-based, convert:
y = y .- minimum(y) .+ 1

N, J = size(distance)
println("Number of households: $N, number of schools: $J")


N, J = size(distance)
println("N: $N, J: $J")

# check dimentions
println("Checking dimensions...")
@assert length(y) == N
@assert length(test_scores) == J
@assert length(sports) == J
@assert size(distance, 1) == N
@assert size(distance, 2) == J

### 4.2
println("--------------------------------")
println("Problem 2.4")
println("Estimate the plain logit model by maximimum likelihood without the xi_j parameters")
println("--------------------------------")

function loglik_joint_full_noxi(params::AbstractVector,
                           test_scores::AbstractVector,
                           sports::AbstractVector,
                           distance::AbstractMatrix,
                           y::AbstractVector{<:Integer})
    T = eltype(params)           # Dual-friendly element type
    N = length(y)
    J = length(test_scores)

    alpha = params[1]
    beta1, beta2 = params[2], params[3]

    U = Array{T}(undef, N, J)    # not Float64
    @inbounds for i in 1:N, j in 1:J
        U[i, j] = beta1*test_scores[j] + beta2*sports[j] - alpha*distance[i, j]
    end

    LL = zero(T)                 # not 0.0
    @inbounds for i in 1:N
        Ui = @view U[i, :]
        m = maximum(Ui)                                   # stable log-sum-exp
        log_denom = m + log(sum(exp.(Ui .- m)))
        LL += Ui[y[i]] - log_denom
    end
    return -LL   # negative log-likelihood
end

function loglik_grad_full_noxi!(G, params, test_scores, sports, distance, y)
    N = length(y)
    J = length(test_scores)

    alpha = params[1]
    beta1, beta2 = params[2:3]

    # reset gradient
    G[:] .= 0.0

    for i in 1:N
        # utilities
        Ui = [beta1 * test_scores[j] + beta2 * sports[j] - alpha * distance[i, j] for j in 1:J]

        # choice probs
        expU = exp.(Ui .- maximum(Ui))  # stability
        P = expU ./ sum(expU)

        yi = y[i]

        # contributions for each parameter
        # dUi/dalpha = -distance[i,j]
        G[1] += -distance[i, yi] - sum(P[j] * (-distance[i, j]) for j in 1:J)

        # dUi/dbeta1 = test_scores[j]
        G[2] += test_scores[yi] - sum(P[j] * test_scores[j] for j in 1:J)

        # dUi/dbeta2 = sports[j]
        G[3] += sports[yi] - sum(P[j] * sports[j] for j in 1:J)
        
    end

    G .*= -1.0   # because objective is -LL
    return G
end

# Initial values
init_params = zeros(3)
println("Initial parameters: ", init_params)

# MLE estimation
println("Starting optimization...")
f_baseline_noxi(p) = loglik_joint_full_noxi(p, test_scores, sports, distance, y)
g_baseline_noxi!(G, p) = loglik_grad_full_noxi!(G, p, test_scores, sports, distance, y)

obj_baseline = OnceDifferentiable(f_baseline_noxi, g_baseline_noxi!, init_params)
result = optimize(obj_baseline, init_params, LBFGS(), options)

params_hat_baseline_noxi = Optim.minimizer(result)

alpha_hat = params_hat_baseline_noxi[1]
beta_hat = params_hat_baseline_noxi[2:3]
xi_hat = [0.0; params_hat_baseline_noxi[4:end]]   # xi1 = 0

println("----------------------------")
println("MLE Results")
println("----------------------------")
println("Estimated alpha: ", alpha_hat)
println("Estimated beta: ", beta_hat)
println("Estimated xi: ", xi_hat)

# Computing standard errors
hessian = ForwardDiff.hessian(p -> f_baseline_noxi(p), params_hat_baseline_noxi)

# Hessian of the NEGATIVE log-likelihood
H = ForwardDiff.hessian(f_baseline_noxi, params_hat_baseline_noxi)

# Observed Fisher information ≈ H, so:
vcov = inv(H)                 # if f_baseline is NEGATIVE log-lik
se_noxi = sqrt.(diag(vcov))

println("Parameter estimates: ", params_hat_baseline_noxi)
println("Standard errors: ", se_noxi)


### 4 Estimate the plain logit model by maximimum likelihood.
println("--------------------------------")
println("Problem 2.4")
println("Estimate the plain logit model by maximimum likelihood.")
println("--------------------------------")

function loglik_joint_full(params::AbstractVector,
                           test_scores::AbstractVector,
                           sports::AbstractVector,
                           distance::AbstractMatrix,
                           y::AbstractVector{<:Integer})
    T = eltype(params)           # Dual-friendly element type
    N = length(y)
    J = length(test_scores)

    alpha = params[1]
    beta1, beta2 = params[2], params[3]
    xi = vcat(zero(T), params[4:3+J-1])   # normalize ξ₁ = 0

    U = Array{T}(undef, N, J)    # not Float64
    @inbounds for i in 1:N, j in 1:J
        U[i, j] = beta1*test_scores[j] + beta2*sports[j] + xi[j] - alpha*distance[i, j]
    end

    LL = zero(T)                 # not 0.0
    @inbounds for i in 1:N
        Ui = @view U[i, :]
        m = maximum(Ui)                                   # stable log-sum-exp
        log_denom = m + log(sum(exp.(Ui .- m)))
        LL += Ui[y[i]] - log_denom
    end
    return -LL   # negative log-likelihood
end

function loglik_grad_full!(G, params, test_scores, sports, distance, y)
    N = length(y)
    J = length(test_scores)

    alpha = params[1]
    beta1, beta2 = params[2:3]
    xi = [0.0; params[4:3+J-1]]   # xi1 = 0 for normalization

    # reset gradient
    G[:] .= 0.0

    for i in 1:N
        # utilities
        Ui = [beta1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j] for j in 1:J]

        # choice probs
        expU = exp.(Ui .- maximum(Ui))  # stability
        P = expU ./ sum(expU)

        yi = y[i]

        # contributions for each parameter
        # dUi/dalpha = -distance[i,j]
        G[1] += -distance[i, yi] - sum(P[j] * (-distance[i, j]) for j in 1:J)

        # dUi/dbeta1 = test_scores[j]
        G[2] += test_scores[yi] - sum(P[j] * test_scores[j] for j in 1:J)

        # dUi/dbeta2 = sports[j]
        G[3] += sports[yi] - sum(P[j] * sports[j] for j in 1:J)

        # dUi/dxi_m = indicator(j==m), for m=2..J
        for m in 2:J
            G[3+m-1] += (yi == m ? 1.0 : 0.0) - P[m]
        end
    end

    G .*= -1.0   # because objective is -LL
    return G
end

# Initial values
init_params = zeros(3 + J - 1)
println("Initial parameters: ", init_params)

# MLE estimation
println("Starting optimization...")
f_baseline(p) = loglik_joint_full(p, test_scores, sports, distance, y)
g_baseline!(G, p) = loglik_grad_full!(G, p, test_scores, sports, distance, y)

obj_baseline = OnceDifferentiable(f_baseline, g_baseline!, init_params)
result = optimize(obj_baseline, init_params, LBFGS(), options)

params_hat_baseline = Optim.minimizer(result)

alpha_hat = params_hat_baseline[1]
beta_hat = params_hat_baseline[2:3]
xi_hat = [0.0; params_hat_baseline[4:end]]   # xi1 = 0

println("----------------------------")
println("MLE Results")
println("----------------------------")
println("Estimated alpha: ", alpha_hat)
println("Estimated beta: ", beta_hat)
println("Estimated xi: ", xi_hat)

# Hessian of the NEGATIVE log-likelihood
H = ForwardDiff.hessian(f_baseline, params_hat_baseline)
H = Symmetric((H + H') / 2) 
# Observed Fisher information ≈ H, so:
vcov = inv(H)                 # if f_baseline is NEGATIVE log-lik
se   = sqrt.(diag(vcov))

println("Parameter estimates: ", params_hat_baseline)
println("Standard errors:     ", se)

# -------------------- Latex result --------------------
parameter_xi = ["\$\\xi_$j\$" for j in 1:J]

# Combine with other parameters
parameters = vcat([raw"$\alpha$",
                   raw"$\beta_1$",
                   raw"$\beta_2$"], parameter_xi)

vals1 = vcat([alpha_hat, beta_hat[1], beta_hat[2]], xi_hat)  # concatenate values

df1 = DataFrame(
    Parameter=parameters,
    Estimate=round.(vals1, digits=6)
)

tab1 = String(latexify(df1, env=:tabular, latex=false))

latex_table1 = """
\\begin{table}[H]
\\centering
$tab1
\\caption{Parameter estimates for the school full choice model.}
\\end{table}
"""

open(joinpath(latex_dir, "table_4.tex"), "w") do f
    write(f, latex_table1)
end

println("Table saved to latex/table_4.tex")






### 5
println("--------------------------------")
println("Problem 2.5")
println("Estimate the restricted logit model by maximimum likelihood.")
println("--------------------------------")
function loglik_joint_restricted(params, y)
    N = length(y)
    J = length(params) + 1
    xi = [0.0; params[1:end]]       # xi1 = 0 for normalization

    U = Array{Float64}(undef, N, J)
    for i in 1:N
        for j in 1:J
            U[i, j] = xi[j]
        end
    end

    LL = 0.0
    for i in 1:N
        log_denom = log(sum(exp.(U[i, :])))
        LL += U[i, y[i]] - log_denom
    end
    return -LL
end

# Initial values
init_params = zeros(J - 1)

# MLE estimation
result = optimize(
    p -> loglik_joint_restricted(p, y),
    init_params,
    LBFGS(),
    options
)

params_hat = Optim.minimizer(result)
xi_hat = [0.0; params_hat[1:end]]   # xi1 = 0

println("----------------------------")
println("MLE Results")
println(raw"----------------------------")
println("Estimated xi: ", xi_hat)

# -------------------- Latex result --------------------
parameter_xi = ["\$\\xi_$j\$" for j in 1:J]

# Combine with other parameters
parameters = vcat([raw"$\alpha$",
                   raw"$\beta_1$",
                   raw"$\beta_2$"], parameter_xi)


df2 = DataFrame(
    Parameter=parameter_xi,
    Estimate=round.(xi_hat, digits=6)
)

tab2 = String(latexify(df2, env=:tabular, latex=false))

latex_table2 = """
\\begin{table}[H]
\\centering
$tab2
\\caption{Parameter estimates for the school restricted choice model.}
\\end{table}
"""



open(joinpath(latex_dir, "table_5.tex"), "w") do f
    write(f, latex_table2)
end

println("Table saved to latex/table_5.tex")

break

### 6
# Latex

### 7
println("--------------------------------")
println("Problem 2.7")
println("Estimating the logit model with simulation using montecarlo methods...")
println("--------------------------------")

params_hat_baseline = [
    0.201097,
    -0.211697,
    0.20147,
    -0.632777,
    -1.030061,
    0.364199,
    -0.897948
]  # from previous estimation  


options = Optim.Options(
    g_tol=1e-6,          # gradient tolerance
    iterations=40,       # max iterations
    outer_iterations = 10,
    show_trace=true,
    show_every=10       # print trace every 100 iterations
)

function loglik_joint_simulated_MC(params, test_scores, sports, distance, y, R)
    N, J = size(distance)

    alpha = params[1]
    beta1_mu = params[2]
    beta2 = params[3]
    xi = [0.0; params[4:3+J-1]]
    sigma_b = params[3+J]

    # simulate random coefficients
    beta1_draws = rand(Normal(beta1_mu, sigma_b), R)

    LL = 0.0
    for i in 1:N
        prob_sim = zeros(J)
        for r in 1:R
            beta1 = beta1_draws[r]
            U = [beta1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j] for j in 1:J]
            expU = exp.(U .- maximum(U)) # numerical stability
            P = expU ./ sum(expU)
            prob_sim .+= P
        end
        prob_sim ./= R  # average across R

        LL += log(prob_sim[y[i]])
    end

    return -LL
end

# Initial values
init_params = vcat(params_hat_baseline, 1)
println("Initial parameters: ", init_params)

# MLE estimation
println("Starting optimization...")
R = 80
lb = fill(-Inf, length(init_params))
lb[end] = 1e-6         # last parameter > 0

ub = fill(Inf, length(init_params))  # no upper bounds


function loglik_grad!(G, params, test_scores, sports, distance, y, R)
    N, J = size(distance)

    alpha = params[1]
    beta1_mu = params[2]
    beta2 = params[3]
    xi = [0.0; params[4:3+J-1]]
    sigma_b = params[3+J]

    R = size(R, 1)
    G[:] .= 0.0

    for i in 1:N
        prob_sim = zeros(J)
        dprob_sim = zeros(length(params), J)
        eps_draws = randn(R)  # standard normal draws

        for r in 1:R
            beta1 = beta1_mu + sigma_b * eps_draws[r]
            U = [beta1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j] for j in 1:J]
            expU = exp.(U .- maximum(U))
            P = expU ./ sum(expU)

            prob_sim .+= P

            # derivatives wrt utilities
            for m in 1:J
                dU = zeros(length(params))
                dU[1] = -distance[i, m]
                dU[2] = test_scores[m]
                dU[3] = sports[m]
                if m > 1
                    dU[3+m-1] = 1.0
                end
                dU[end] = R[r] * test_scores[m]

                for j in 1:J
                    coeff = P[j] * ((j == m) - P[m])
                    dprob_sim[:, j] .+= coeff .* dU
                end
            end
        end

        prob_sim ./= R
        dprob_sim ./= R

        yi = y[i]
        G .+= dprob_sim[:, yi] ./ prob_sim[yi]
    end

    G .*= -1.0   # because objective is -LL
end

f_simulated_MC(p) = loglik_joint_simulated_MC(p, test_scores, sports, distance, y, R)
g_simulated_MC!(G, p) = loglik_grad!(G, p, test_scores, sports, distance, y, R)

obj_simulated_MC = OnceDifferentiable(f_simulated_MC, g_simulated_MC!, init_params)
result = optimize(obj_simulated_MC, lb, ub, init_params, Fminbox(LBFGS()), options)

params_hat = Optim.minimizer(result)



params_hat = Optim.minimizer(result)
println("Estimated parameters with constraint: ", params_hat)

params_hat = Optim.minimizer(result)
alpha_hat = params_hat[1]
beta_hat = params_hat[2:3]
xi_hat = [0.0; params_hat[4:end-1]]   # xi1 = 0
sigma_b_hat = params_hat[end]

println("----------------------------")
println("MLE Results")
println("----------------------------")
println("Estimated alpha: ", alpha_hat)
println("Estimated beta: ", beta_hat)
println("Estimated xi: ", xi_hat)

# -------------------- Latex result --------------------
parameter_xi = ["\$\\xi_$j\$" for j in 1:J]

# Combine with other parameters
parameters = vcat([raw"$\alpha$",
                   raw"$\beta_1$",
                   raw"$\beta_2$"], parameter_xi, [raw"$\sigma_b$"])

vals1 = vcat([alpha_hat, beta_hat[1], beta_hat[2]], xi_hat, [sigma_b_hat])

df1 = DataFrame(
    Parameter=parameters,
    Estimate=round.(vals1, digits=6)
)

tab1 = String(latexify(df1, env=:tabular, latex=false))

latex_table1 = """
\\begin{table}[H]
\\centering
$tab1
\\caption{Parameter estimates for the school full simulated choice model using Monte Carlo methods.}
\\end{table}
"""
open(joinpath(latex_dir, "table_7_1.tex"), "w") do f
    write(f, latex_table1)
end

println("Table saved to latex/table_7_1.tex")
println(latex_table1)

println("--------------------------------")
println("Estimating the logit model with simulation using gaussian Hermite quadrature...")
println("--------------------------------")

function loglik_joint_simulated_GH(params, test_scores, sports, distance, y, k)
    N, J = size(distance)

    alpha = params[1]
    beta1_mu = params[2]
    beta2 = params[3]
    xi = [0.0; params[4:3+J-1]]   # normalize xi1 = 0
    sigma_b = params[3+J]

    # quadrature nodes and weights
    x, w = gausshermite(k)
    nodes = beta1_mu .+ sigma_b .* sqrt(2) .* x
    weights = w ./ sqrt(pi)

    LL = 0.0
    for i in 1:N
        prob_sim = zeros(J)
        for m in 1:k
            beta1 = nodes[m]
            weight = weights[m]
            U = [beta1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j] for j in 1:J]
            expU = exp.(U .- maximum(U))
            P = expU ./ sum(expU)
            prob_sim .+= weight .* P
        end
        LL += log(prob_sim[y[i]])
    end

    return -LL
end


function loglik_grad_GH!(G, params, test_scores, sports, distance, y, k)
    N, J = size(distance)
    nparams = length(params)    # should be 3 + J

    # unpack
    alpha = params[1]
    beta1_mu = params[2]
    beta2 = params[3]
    xi = [0.0; params[4:3+J-1]]   # xi[1]=0
    sigma_b = params[3+J]

    # Gauss-Hermite nodes & weights
    x, w = gausshermite(k)                 # x: nodes, w: weights for ∫ e^{-x^2} f(x) dx
    nodes = beta1_mu .+ sigma_b .* sqrt(2.0) .* x
    weights = w ./ sqrt(pi)                # transformation to normal integrand

    # zero gradient
    G[:] .= 0.0

    # loop observations
    for i in 1:N
        # accumulate probability (mixture over nodes)
        prob_mix = zeros(J)
        # we will accumulate numerator for gradient: sum_m weight_m * P_m * (dU_y - E_P dU)
        grad_num = zeros(nparams)

        for m in 1:k
            β1 = nodes[m]
            weight = weights[m]

            # utilities for node m
            U = similar(prob_mix)
            for j in 1:J
                U[j] = β1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j]
            end

            # softmax probabilities (stable)
            expU = exp.(U .- maximum(U))
            P = expU ./ sum(expU)

            prob_mix .+= weight .* P

            # compute dU_j (vector of length nparams) for each j, but we only need E_P[dU] and dU_{y}
            # compute E_P_dU = sum_j P[j] * dU_j
            E_P_dU = zeros(nparams)

            # we also store dU for the chosen alternative later (or compute on the fly)
            dU_for_j = Vector{Float64}(undef, nparams)  # temporary

            for j in 1:J
                # build dU_j
                fill!(dU_for_j, 0.0)
                dU_for_j[1] = -distance[i, j]            # dU/dalpha
                dU_for_j[2] = test_scores[j]            # dU/dbeta1_mu
                dU_for_j[3] = sports[j]                 # dU/dbeta2
                if j > 1
                    xi_index = 3 + (j - 1)              # param index for xi_j
                    dU_for_j[xi_index] = 1.0
                end
                dU_for_j[end] = sqrt(2.0) * x[m] * test_scores[j]  # dU/dsigma_b

                # accumulate expectation under P
                @inbounds for q = 1:nparams
                    E_P_dU[q] += P[j] * dU_for_j[q]
                end
            end

            # dU for chosen alt y_i (recompute dU_y to avoid storing whole matrix)
            yi = y[i]
            dU_y = zeros(nparams)
            dU_y[1] = -distance[i, yi]
            dU_y[2] = test_scores[yi]
            dU_y[3] = sports[yi]
            if yi > 1
                dU_y[3+(yi-1)] = 1.0
            end
            dU_y[end] = sqrt(2.0) * x[m] * test_scores[yi]

            # derivative of P_y wrt params at node m:
            # dP_y = P[yi] * (dU_y - E_P_dU)
            coeff = P[yi]
            @inbounds for q = 1:nparams
                grad_num[q] += weight * (coeff * (dU_y[q] - E_P_dU[q]))
            end
        end

        # now grad contribution from observation i:
        # d log(prob_mix[yi]) = grad_num / prob_mix[yi]
        pi_y = prob_mix[y[i]]
        # safety: if pi_y extremely small, it's numerically unstable; user can handle/report later
        @inbounds for q = 1:nparams
            G[q] += grad_num[q] / pi_y
        end
    end

    # we computed gradient of LL; the objective returns -LL, so set G = - (dLL)
    G .*= -1.0
    return nothing
end



init_params = vcat(params_hat_baseline, 1)
println("Initial parameters: ", init_params)
# ----------------------------
# MLE estimation
# ----------------------------
println("Starting optimization...")
k = 10
lb = fill(-Inf, length(init_params))
lb[end] = 1e-6         # last parameter > 0

ub = fill(Inf, length(init_params))  # no upper bounds

# Use Fminbox to apply bounds
# objective as you posted
f_simulated_GH(p) = loglik_joint_simulated_GH(p, test_scores, sports, distance, y, k)

# gradient wrapper for OnceDifferentiable
g_simulated_GH!(G, p) = loglik_grad_GH!(G, p, test_scores, sports, distance, y, k)

obj_simulated_GH = OnceDifferentiable(f_simulated_GH, g_simulated_GH!, init_params)
result = optimize(obj_simulated_GH, lb, ub, init_params, Fminbox(LBFGS()), options)

params_hat = Optim.minimizer(result)


println("Estimated parameters with constraint: ", params_hat)


params_hat = Optim.minimizer(result)
alpha_hat = params_hat[1]
beta_hat = params_hat[2:3]
xi_hat = [0.0; params_hat[4:end-1]]   # xi1 = 0
sigma_b_hat = params_hat[end]

println("----------------------------")
println("MLE Results")
println("----------------------------")
println("Estimated alpha: ", alpha_hat)
println("Estimated beta: ", beta_hat)
println("Estimated xi: ", xi_hat)

# -------------------- Latex result --------------------
parameter_xi = ["\$\\xi_$j\$" for j in 1:J]

# Combine with other parameters
parameters = vcat([raw"$\alpha$",
                   raw"$\beta_1$",
                   raw"$\beta_2$"], parameter_xi, [raw"$\sigma_b$"])
vals1 = vcat([alpha_hat, beta_hat[1], beta_hat[2]], xi_hat, [sigma_b_hat])

df1 = DataFrame(
    Parameter=parameters,
    Estimate=round.(vals1, digits=6)
)

tab1 = String(latexify(df1, env=:tabular, latex=false))

latex_table1 = """
\\begin{table}[H]
\\centering
$tab1
\\caption{Parameter estimates for the school full simulated choice model using Gaussian Quadrature methods.}
\\end{table}
"""

open(joinpath(latex_dir, "table_7_2.tex"), "w") do f
    write(f, latex_table1)
end

println("Table saved to latex/table_7_2.tex")
println(latex_table1)



### 10
println("--------------------------------")
println("Problem 2.10")
println("Computing the estimates using methods of moment...")
println("--------------------------------")



function msm_moments(params, test_scores, sports, distance, y, Q)
    N, J = size(distance)
    alpha = params[1]
    beta1_mu, beta2 = params[2:3]
    xi = [0.0; params[4:3+J-1]]   # xi1 = 0 for normalization
    sigma_b = params[3+J]

    # Gauss-Hermite quadrature points and weights
    nodes, weights = gausshermite(Q)
    # scale nodes for Normal(beta1_mu, sigma_b)
    nodes = sqrt(2) * sigma_b * nodes .+ beta1_mu
    weights = weights ./ sqrt(pi)  # weights sum to 1

    # initialize probabilities
    P = zeros(N, J)

    for i in 1:N
        P_i = zeros(J)
        for q in 1:Q
            β1 = nodes[q]
            w = weights[q]
            U = [β1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j] for j in 1:J]
            expU = exp.(U .- maximum(U))
            P_i .+= w .* (expU ./ sum(expU))
        end
        P[i, :] .= P_i
    end

    # compute moments
    g = zeros(N, J * 3)
    for i in 1:N
        for j in 1:J
            z_ij = [test_scores[j], sports[j], distance[i, j]]
            g[i, (3*(j-1)+1):(3*j)] .= (Int(y[i] == j) - P[i, j]) .* z_ij
        end
    end

    return g
end

function msm_grad!(G, params, test_scores, sports, distance, y, Q)
    N, J = size(distance)
    nparams = length(params)

    alpha = params[1]
    beta1_mu, beta2 = params[2:3]
    xi = [0.0; params[4:3+J-1]]
    sigma_b = params[3+J]

    nodes, weights = gausshermite(Q)
    nodes = sqrt(2) * sigma_b * nodes .+ beta1_mu
    weights = weights ./ sqrt(pi)

    g_matrix = zeros(N, J * 3)
    J_g = zeros(N * J * 3, nparams)

    for i in 1:N
        P_i = zeros(J)
        dP_dθ = zeros(J, nparams)

        for q in 1:Q
            β1 = nodes[q]
            w = weights[q]

            U = [β1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j] for j in 1:J]
            expU = exp.(U .- maximum(U))
            P = expU ./ sum(expU)

            dU = [zeros(nparams) for j in 1:J]
            for j in 1:J
                dU[j][1] = -distance[i, j]           # ∂U/∂α
                dU[j][2] = test_scores[j]            # ∂U/∂β1_mu
                dU[j][3] = sports[j]                  # ∂U/∂β2
                if j > 1
                    dU[j][3+(j-1)] = 1.0             # xi_j
                end
                dU[j][end] = (β1 - beta1_mu) / sigma_b * test_scores[j]  # ∂/∂σ_b
            end

            E_dU = zeros(nparams)
            for j in 1:J
                E_dU .+= P[j] .* dU[j]
            end

            for j in 1:J
                dP_dθ[j, :] .+= w .* P[j] .* (dU[j] .- E_dU)
            end

            P_i .+= w .* P
        end

        for j in 1:J
            z_ij = [test_scores[j], sports[j], distance[i, j]]
            idx = (3*(j-1)+1):(3*j)
            g_matrix[i, idx] .= (Int(y[i] == j) - P_i[j]) .* z_ij

            for q in 1:nparams
                J_g[idx, q] .+= -z_ij .* dP_dθ[j, q]
            end
        end
    end

    g_vec = vec(g_matrix)
    G[:] = 2 .* (J_g' * g_vec)
    return nothing
end



function msm_objective(params, test_scores, sports, distance, y, k)
    g_matrix = msm_moments(params, test_scores, sports, distance, y, k)
    g_vec = vec(g_matrix)      # flatten into 1D vector
    return dot(g_vec, g_vec)   # now works
end

# Initial values
init_params = vcat(params_hat_baseline, 1)

# GMM estimation
R = 80
lb = fill(-Inf, length(init_params))
lb[end] = 1e-6         # last parameter > 0

ub = fill(Inf, length(init_params))  # no upper bounds
f_msm(p) = msm_objective(p, test_scores, sports, distance, y, k)
g_msm!(G, p) = msm_grad!(G, p, test_scores, sports, distance, y, k)

obj_msm = OnceDifferentiable(f_msm, g_msm!, init_params)
result = optimize(obj_msm, lb, ub, init_params, Fminbox(LBFGS()), options)



params_hat = Optim.minimizer(result)

alpha_hat = params_hat[1]
beta_hat = params_hat[2:3]
xi_hat = [0.0; params_hat[4:end-1]]   # xi1 = 0
sigma_b_hat = params_hat[end]

println("----------------------------")
println("MLE Results")
println("----------------------------")
println("Estimated alpha: ", alpha_hat)
println("Estimated beta: ", beta_hat)
println("Estimated xi: ", xi_hat)

# -------------------- Latex result --------------------
parameter_xi = ["\$\\xi_$j\$" for j in 1:J]

# Combine with other parameters
parameters = vcat([raw"$\alpha$",
                   raw"$\beta_1$",
                   raw"$\beta_2$"], parameter_xi, [raw"$\sigma_b$"])
vals1 = vcat([alpha_hat, beta_hat[1], beta_hat[2]], xi_hat, [sigma_b_hat])

df1 = DataFrame(
    Parameter=parameters,
    Estimate=round.(vals1, digits=6)
)

tab1 = String(latexify(df1, env=:tabular, latex=false))

latex_table1 = """
\\begin{table}[H]
\\centering
$tab1
\\caption{Parameter estimates for the school full simulated choice model using Simulated Method of Moments.}
\\end{table}
"""

open(joinpath(latex_dir, "table_10.tex"), "w") do f
    write(f, latex_table1)
end

println("Table saved to latex/table_10.tex")
println(latex_table1)


### 9
# println("--------------------------------")
# println("Problem 2.9")
# println("Computing the Jacobian of the Moments")
# println("--------------------------------")

# function jacobian_msm(params, test_scores, sports, distance, y, R)
#     N, J = size(distance)
#     K = length(params)
#     L = N * J * 3
#     G = zeros(L, K)

#     # For each household, school, instrument
#     idx = 1
#     beta1_mu, beta2 = params[2:3]
#     xi = [0.0; params[4:3+J-1]]
#     alpha = params[1]
#     sigma_b = params[3+J]

#     # Simulate random coefficients
#     beta1_draws = rand(Normal(beta1_mu, sigma_b), R)

#     for i in 1:N
#         P_i = zeros(J)
#         dU_dparams = zeros(J, K)
#         # Compute average choice probabilities and derivatives
#         for r in 1:R
#             beta1 = beta1_draws[r]
#             U = [beta1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j] for j in 1:J]
#             expU = exp.(U .- maximum(U))
#             P_r = expU ./ sum(expU)
#             P_i .+= P_r

#             # Derivatives of U w.r.t. params
#             for j in 1:J
#                 dU = zeros(K)
#                 dU[1] = -distance[i, j]             # alpha
#                 dU[2] = test_scores[j]             # beta1_mu
#                 dU[3] = sports[j]                  # beta2
#                 if j > 1
#                     dU[3+(j-1)] = 1.0  # xi_2,...xi_J
#                 end
#                 # sigma_b derivative can be added if desired
#                 dU_dparams[j, :] .+= dU
#             end
#         end
#         P_i ./= R
#         dU_dparams ./= R

#         # Compute Jacobian entries
#         for j in 1:J
#             z_ij = [test_scores[j], sports[j], distance[i, j]]
#             for k_inst in 1:3  # three instruments
#                 for m in 1:K
#                     G[idx, m] = -z_ij[k_inst] * (dU_dparams[j, m] * P_i[j] - sum(P_i .* dU_dparams[:, m]))
#                 end
#                 idx += 1
#             end
#         end
#     end
#     return G
# end

# jacobian = jacobian_msm(params_hat, test_scores, sports, distance, y, 100)
# println("Jacobian size: ", size(jacobian))




### 11
# println("--------------------------------")
# println("Problem 2.11")
# println("Estimating efficient MSM...")
# println("--------------------------------")

# function msm_objective_efficient(params, test_scores, sports, distance, y, R)
#     g_all = msm_moments(params, test_scores, sports, distance, y, R)  # N x L
#     N, L = size(g_all)

#     # Compute mean of moments across households
#     g_mean = mean(g_all, dims=1)             # 1 x L

#     # Covariance matrix of moments (L x L)
#     Omega = (g_all .- g_mean)' * (g_all .- g_mean) / N
#     W = inv(Omega + 1e-8 * I)                  # add small regularization for numerical stability

#     # Stack moments across households
#     g = vec(sum(g_all, dims=1))              # L x 1

#     return dot(g, W * g)
# end

# # Initial values
# init_params = vcat(params_hat_baseline, 1)

# # GMM estimation
# R = 80
# lb = fill(-Inf, length(init_params))
# lb[end] = 1e-6         # last parameter > 0

# ub = fill(Inf, length(init_params))  # no upper bounds
# result = optimize(
#     p -> msm_objective_efficient(p, test_scores, sports, distance, y, R),
#     lb,
#     ub,
#     init_params,
#     Fminbox(LBFGS()),
#     options
# )


# params_hat = Optim.minimizer(result)
# alpha_hat = params_hat[1]
# beta_hat = params_hat[2:3]
# xi_hat = [0.0; params_hat[4:end-1]]   # xi1 = 0
# sigma_b_hat = params_hat[end]

# println("----------------------------")
# println("MLE Results")
# println("----------------------------")
# println("Estimated alpha: ", alpha_hat)
# println("Estimated beta: ", beta_hat)
# println("Estimated xi: ", xi_hat)

# # -------------------- Latex result --------------------
# parameter_xi = ["\$\\xi_$j\$" for j in 1:J]

# # Combine with other parameters
# parameters = vcat([raw"$\alpha$",
#                    raw"$\beta_1$",
#                    raw"$\beta_2$"], parameter_xi, [raw"$\sigma_b$"])
# vals1 = vcat([alpha_hat, beta_hat[1], beta_hat[2]], xi_hat, [sigma_b_hat])

# df1 = DataFrame(
#     Parameter=parameters,
#     Estimate=round.(vals1, digits=6)
# )

# tab1 = String(latexify(df1, env=:tabular, latex=false))

# latex_table1 = """
# \\begin{table}[H]
# \\centering
# $tab1
# \\caption{Parameter estimates for the school full simulated choice model using Simulated Method of Moments.}
# \\end{table}
# """

# open(joinpath(latex_dir, "table_11.tex"), "w") do f
#     write(f, latex_table1)
# end

# println("Table saved to latex/table_11.tex")
# println(latex_table1)
