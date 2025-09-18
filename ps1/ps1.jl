### Problem Set 1
# Author: Stefano Sperti

using DataFrames, CSV, Plots, Statistics

cd(@__DIR__)
figures_dir = joinpath(@__DIR__, "figures")
if !isdir(figures_dir)
    mkdir(figures_dir)
end

school_dataset = CSV.read("schools_dataset.csv", DataFrame)
println("Number of rows: ", size(school_dataset, 1))
println("Number of columns: ", size(school_dataset, 2))
println("Column names: ", names(school_dataset))
println("First 5 rows: ")
first(school_dataset, 5)


### 1
histogram(school_dataset[!, :distance], bins=20, title="Histogram of Distance to all the Schools", xlabel="Distance from schools School", ylabel="Frequency")
savefig(joinpath(figures_dir, "histogram_distance_all_schools.png"))

min_distances_vector = combine(groupby(school_dataset, :household_id),
                               :distance => minimum).distance_minimum
histogram(min_distances_vector, bins=20, title="Histogram of Minimum Distance to Schools", xlabel="Minimum Distance from schools School", ylabel="Frequency")
savefig(joinpath(figures_dir, "histogram_min_distance_schools.png"))



### 3
using Random, Distributions, Optim, LinearAlgebra

# ----------------------------
# Simulate data
# ----------------------------
Random.seed!(123)


school_info = unique(school_dataset[:, [:school_id, :test_scores, :sports]])
rename!(school_info, [:test_scores, :sports] .=> [:test_scores_school, :sports_school])

# Sort by school_id to match pivoted matrices
school_info = sort(school_info, :school_id)

households = sort(unique(school_dataset.household_id))
schools = sort(unique(school_dataset.school_id))  # this will be the row order

y_df = unstack(school_dataset, :school_id, :household_id, :y_ij)
# Ensure rows are in the correct school order
y_df = sort(y_df, :school_id)
# Convert to matrix
y = Matrix(y_df[:, Not(:school_id)])
y = y .+ 1

# ----------------------------
# Pivot distance
# ----------------------------
dist_df = unstack(school_dataset, :school_id, :household_id, :distance)
dist_df = sort(dist_df, :school_id)
distance = Matrix(dist_df[:, Not(:school_id)])

N, J = size(distance)

# Simulate choices
function utilities_matrix(β, ξ, test_scores, sports, distance)
    N, J = size(distance)
    U = Array{Float64}(undef, N, J)
    β1, β2, α = β
    for i in 1:N
        for j in 1:J
            U[i,j] = β1*test_scores[j] + β2*sports[j] + ξ[j] - α*distance[i,j]
        end
    end
    return U
end
# ----------------------------
# Log-likelihood function
# ----------------------------
function loglik_joint(params, test_scores, sports, distance, y)
    # params = [β1, β2, α, ξ2, ξ3, ..., ξJ] (ξ1=0)
    N, J = size(distance)
    β = params[1:3]
    ξ = [0.0; params[4:end]]       # ξ1 = 0 for normalization

    U = utilities_matrix(β, ξ, test_scores, sports, distance)

    ℓ = 0.0
    for i in 1:N
        log_denom = log(sum(exp.(U[i,:])))
        ℓ += U[i, y[i]] - log_denom
    end
    return ℓ
end

nll_joint(params, test_scores, sports, distance, y) = -loglik_joint(params, test_scores, sports, distance, y)

# ----------------------------
# Initial values
# ----------------------------
init_params = zeros(3 + J - 1)   # β1, β2, α, ξ2,...,ξJ

# ----------------------------
# MLE estimation
# ----------------------------
result = optimize(
    p -> nll_joint(p, test_scores, sports, distance, y),
    init_params,
    LBFGS()
)

params_hat = Optim.minimizer(result)
β_hat = params_hat[1:3]
ξ_hat = [0.0; params_hat[4:end]]   # ξ1 = 0

println("Estimated β: ", β_hat)
println("Estimated ξ: ", ξ_hat)
