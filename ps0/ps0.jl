### Problem Set 0
# Author: Stefano Sperti

### Part 0:
# 1
# solved in pdf.

using Random
Random.seed!(1234)

println("--------------------------------")
println("Problem 0.2")
println("--------------------------------")
x = randn(1000) * 10000
IV_basic = log(sum(exp.(x)))
println("Basic: $IV_basic") 

# 2
max = maximum(x)
IV_max= log(sum(exp.(x .- max))) + max
println("Deleting the maximum: $IV_max")

# 3
println("--------------------------------")
println("Problem 0.3")
println("--------------------------------")
using StatsFuns
IV_statsfuns = logsumexp(x)
println("StatsFuns: $IV_statsfuns")

## Part 1:
# 1
println("--------------------------------")
println("Problem 1.1")
println("--------------------------------")
using LinearAlgebra
P =[0.2 0.4 0.4; 
    0.1 0.3 0.6; 
    0.5 0.1 0.4]

P100 = P^100

"""
Compute the ergodic distribution of a stochastic matrix P.
"""
function ergodic_distribution(P)
    if !isapprox(sum(P, dims=2), ones(size(P, 1)))
        error("P is not a stochastic matrix")
    end
    n = size(P, 1)
    A = [P' - I; ones(1, n)]
    b = [zeros(n); 1]
    π = A \ b
    return π
end

π = ergodic_distribution(P)
println("Ergodic distribution: $π")
println("P^100: $P100")


# Part 2
# 1
println("--------------------------------")
println("Problem 2.1")
println("--------------------------------")

binomiallogit(t) = t ≥ 0 ? inv(1 + exp(-t)) : exp(t) / (1 + exp(t))

# 2
println("--------------------------------")
println("Problem 2.2")
println("--------------------------------")
using QuadGK, Distributions
mu = 0.5
sigma = 2
normal_pdf = x -> pdf(Normal(mu, sigma), x)
X = 0.5

m = dot(X, mu)
v = dot(X, sigma * X)             # X'sigmaX
s = sqrt(v)
f(t) =  binomiallogit(t)  * pdf(Normal(m, s), t)
quad_1, err = quadgk(f, -Inf, Inf; rtol=1e-14, atol=1e-14)


# 3
println("--------------------------------")
println("Problem 2.3")
println("--------------------------------")

function montecarlo_integration(f, distribution, n)
    samples = rand(distribution, n)
    integral = mean(f.(samples))
    return integral
end

mean200_1 = montecarlo_integration(t -> binomiallogit(t), Normal(m, s), 200)
mean400_1 = montecarlo_integration(t -> binomiallogit(t), Normal(m, s), 400)


# 4 Gauss Hermite
println("--------------------------------")
println("Problem 2.4")
println("--------------------------------")
using FastGaussQuadrature
function gh_quadrature(k; X, mu, sigma)
    # mean along X
    m = if mu isa Number && X isa Number
        X * mu
    elseif mu isa AbstractVector && X isa AbstractVector
        @assert length(mu) == length(X)
        dot(X, mu)
    else
        throw(ArgumentError("X and mu must both be scalars or both vectors of equal length"))
    end

    # variance along X
    s2 = if sigma isa Number
        # isotropic sigma = sigma² I  (also covers 1D case)
        (X isa Number) ? (X^2) * sigma : sigma * sum(abs2, X)
    elseif sigma isa AbstractVector
        @assert X isa AbstractVector && length(sigma) == length(X)
        sum((X.^2) .* sigma)              # diag covariance
    elseif sigma isa AbstractMatrix
        @assert X isa AbstractVector && size(sigma,1) == length(X) == size(sigma,2)
        dot(X, sigma * X)                 # full covariance
    else
        throw(ArgumentError("sigma must be Number, Vector, or square Matrix"))
    end
    s = sqrt(s2)

    # GH nodes/weights (weighting exp(-x^2))
    x, w = gausshermite(k)
    nodes   = m .+ s .* sqrt(2) .* x
    weights = w ./ sqrt(pi)

    return dot(weights, binomiallogit.(nodes))
end

GH4   = gh_quadrature(4, X=X, mu=mu, sigma=sigma)
GH12  = gh_quadrature(12, X=X, mu=mu, sigma=sigma)
GH8   = gh_quadrature(8, X=X, mu=mu, sigma=sigma) #odd one


# 5 compare results
println("--------------------------------")
println("Problem 2.5")
println("--------------------------------")
println("Quadgk: $quad_1")
println("MC 200: $mean200_1")
println("MC 400: $mean400_1")
println("GH 4: $GH4")
println("GH 8: $GH8")
println("GH 12: $GH12") 


# 6 two dimentions
println("--------------------------------")
println("Problem 2.6")
println("--------------------------------")

# --- Example (your numbers) ---
mu    = [0.5, 1.0]
sigma     = [2.0 0.0; 0.0 1.0]
X     = [0.5, 1.0]

m = dot(X, mu)
v = dot(X, sigma * X)           
s = sqrt(v)
f(t) = binomiallogit(t) * pdf(Normal(m, s), t)
quad_2, err =quadgk(f, -Inf, Inf; rtol=1e-14, atol=1e-14)
println("Integral ≈ $quad_2 (quad err ≈ $err)")

mean200_2 = montecarlo_integration(t -> binomiallogit(t), Normal(m, s), 200)
mean400_2 = montecarlo_integration(t -> binomiallogit(t), Normal(m, s), 400)

# 7 create latex table
println("--------------------------------")
println("Problem 2.7")
println("--------------------------------")

using DataFrames, Latexify, Printf

# helper: 6-decimal strings; blank for missing
fmt6(x) = ismissing(x) ? "" : @sprintf("%.6f", x)

using DataFrames, Latexify

# -------------------- 1D --------------------
methods1 = ["QuadGK, True Value", "MC 200", "MC 400", "GH 4", "GH 8", "GH 12"]
vals1    = [quad_1, mean200_1, mean400_1, GH4, GH8, GH12]
npoints1 = ["—", 200, 400, 4, 8, 12]

abs_err1 = [abs(v - quad_1) for v in vals1]
rel_err1 = [iszero(quad_1) ? missing : abs(v - quad_1) / abs(quad_1) * 100 for v in vals1]

abs_err1[1] = 0.0

rel_err1[1] = 0.0

df1 = DataFrame(
    :Method => methods1,
    :Value => round.(vals1, digits=6),
    Symbol("Abs. error") => round.(abs_err1, digits=6),
    Symbol("Rel. error") => round.(rel_err1, digits=6),
    Symbol("N points") => npoints1
)

tab1 = String(latexify(df1, env=:tabular, latex=false))
latex_table1 = """
\\begin{table}[htbp]
\\centering
$tab1
\\caption{1D integration results: value, error, and number of points.}
\\label{tab:results-1d}
\\end{table}
"""
println(latex_table1)

# -------------------- 2D --------------------
methods2 = ["QuadGK, True Value", "MC 200", "MC 400"]
vals2    = [quad_2, mean200_2, mean400_2]
npoints2 = ["—", 200, 400]

abs_err2 = [abs(v - quad_2) for v in vals2]
rel_err2 = [iszero(quad_2) ? missing : abs(v - quad_2) / abs(quad_2) * 100 for v in vals2]

abs_err2[1] = 0.0

rel_err2[1] = 0.0

df2 = DataFrame(
    :Method => methods2,
    :Value => round.(vals2, digits=6),
    Symbol("Abs. error") => round.(abs_err2, digits=6),
    Symbol("Rel. error") => round.(rel_err2, digits=6),
    Symbol("N points") => npoints2
)

tab2 = String(latexify(df2, env=:tabular, latex=false))
latex_table2 = """
\\begin{table}[htbp]
\\centering
$tab2
\\caption{2D integration results: value, error, and number of points.}
\\label{tab:results-2d}
\\end{table}
"""
println(latex_table2)



# 8 binomiallogitmixture
println("--------------------------------")
println("Problem 2.8")
println("--------------------------------")
function binomiallogitmixture(X, mu, sigma, methods)
    m = dot(X, mu)
    v = dot(X, sigma * X)           
    s = sqrt(v)
    if methods == :quadgk
        f(t) = binomiallogit(t) * pdf(Normal(m, s), t)
        integral, err = quadgk(f, -Inf, Inf; rtol=1e-8, atol=1e-8)
        return integral
    elseif methods == :montecarlo
        return montecarlo_integration(t -> binomiallogit(t), Normal(m, s), 1000)
    elseif methods == :gh
        return gh_quadrature(12, X=X, mu=mu, sigma=sigma)
    else
        error("Unknown method: $methods")
    end
end
    
methods = [:quadgk, :montecarlo, :gh]
results = Dict()
for method in methods
    results[method] = binomiallogitmixture(X, mu, sigma, method)
end 
println("Results: $results")
