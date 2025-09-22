### Problem Set 1
# Author: Stefano Sperti

using DataFrames, CSV, Plots, Statistics, Random, Distributions, Optim, LinearAlgebra, DataFrames, Latexify, FastGaussQuadrature

cd(@__DIR__)
figures_dir = joinpath(@__DIR__, "figures")
if !isdir(figures_dir)
    mkdir(figures_dir)
end
open("log_result.txt", "w") do f
    redirect_stdout(f) do

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

        ###
        println("Preparing data for estimation...")
        println("Number of unique households: ", length(unique(school_dataset[:, :household_id])))
        println("Number of unique schools: ", length(unique(school_dataset[:, :school_id])))

        # School characteristics
        school_info = unique(school_dataset[:, [:school_id, :test_scores, :sports]])
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
        ### 4 Estimate the plain logit model by maximimum likelihood.
        println("--------------------------------")
        println("Problem 2.4")
        println("Estimate the plain logit model by maximimum likelihood.")
        println("--------------------------------")

        function loglik_joint_full(params, test_scores, sports, distance, y)
            N = length(y)
            J = length(test_scores)

            alpha = params[1]
            beta1, beta2 = params[2:3]
            xi = [0.0; params[4:3+J-1]]   # xi1 = 0 for normalization

            U = zeros(N, J)
            for i in 1:N
                for j in 1:J
                    U[i, j] = beta1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j]
                end
            end

            LL = 0.0
            for i in 1:N
                Ui = U[i, :]
                m = maximum(Ui)
                log_denom = m + log(sum(exp.(Ui .- m)))  # log-sum-exp
                LL += U[i, y[i]] - log_denom
            end
            return -LL
        end

        # Initial values
        init_params = zeros(3 + J - 1)
        println("Initial parameters: ", init_params)

        # MLE estimation
        println("Starting optimization...")
        result = optimize(
            p -> loglik_joint_full(p, test_scores, sports, distance, y),
            init_params,
            BFGS()
        )

        params_hat = Optim.minimizer(result)
        alpha_hat = params_hat[1]
        beta_hat = params_hat[2:3]
        xi_hat = [0.0; params_hat[4:end]]   # xi1 = 0

        println("----------------------------")
        println("MLE Results")
        println("----------------------------")
        println("Estimated alpha: ", alpha_hat)
        println("Estimated beta: ", beta_hat)
        println("Estimated xi: ", xi_hat)


        # -------------------- Latex result --------------------
        parameter_xi = ["xi_$j" for j in 1:J]  # creates ["xi_1", "xi_2", ..., "xi_J"]

        parameters = vcat(["alpha", "beta1", "beta2"], parameter_xi)  # concatenate arrays

        vals1 = vcat([alpha_hat, beta_hat[1], beta_hat[2]], xi_hat)  # concatenate values

        df1 = DataFrame(
            Parameter=parameters,
            Estimate=round.(vals1, digits=6)
        )

        tab1 = String(latexify(df1, env=:tabular, latex=false))

        latex_table1 = """
        \\begin{table}[htbp]
        \\centering
        $tab1
        \\caption{Parameter estimates for the school full choice model.}
        \\end{table}
        """

        println(latex_table1)

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
            BFGS()
        )

        params_hat = Optim.minimizer(result)
        xi_hat = [0.0; params_hat[1:end]]   # xi1 = 0

        println("----------------------------")
        println("MLE Results")
        println(raw"----------------------------")
        println("Estimated xi: ", xi_hat)

        # -------------------- Latex result --------------------
        parameter_xi = ["xi_$j" for j in 1:J]  # creates ["xi_1", "xi_2", ..., "xi_J"]


        df2 = DataFrame(
            Parameter=parameter_xi,
            Estimate=round.(xi_hat, digits=6)
        )

        tab2 = String(latexify(df2, env=:tabular, latex=false))

        latex_table2 = """
        \\begin{table}[htbp]
        \\centering
        $tab2
        \\caption{Parameter estimates for the school restricted choice model.}
        \\end{table}
        """

        println(latex_table2)


        ### 6
        # Latex

        ### 7
        println("--------------------------------")
        println("Problem 2.7")
        println("Estimating the logit model with simulation using montecarlo methods...")
        println("--------------------------------")
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
                prob_sim ./= R  # average across draws

                LL += log(prob_sim[y[i]])
            end

            return -LL
        end

        # Initial values
        init_params = vcat(zeros(3 + J - 1), 1)
        println("Initial parameters: ", init_params)

        # MLE estimation
        println("Starting optimization...")
        R = 100
        lb = fill(-Inf, length(init_params))
        lb[end] = 1e-6         # last parameter > 0

        ub = fill(Inf, length(init_params))  # no upper bounds

        # Use Fminbox to apply bounds
        result = optimize(
            p -> loglik_joint_simulated_MC(p, test_scores, sports, distance, y, R),
            lb,
            ub,
            init_params,
            Fminbox(BFGS())
        )

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
        parameter_xi = ["xi_$j" for j in 1:J]  # creates ["xi_1", "xi_2", ..., "xi_J"]

        parameters = vcat(["alpha", "beta1", "beta2"], parameter_xi, ["sigma_b"])
        vals1 = vcat([alpha_hat, beta_hat[1], beta_hat[2]], xi_hat, [sigma_b_hat])

        df1 = DataFrame(
            Parameter=parameters,
            Estimate=round.(vals1, digits=6)
        )

        tab1 = String(latexify(df1, env=:tabular, latex=false))

        latex_table1 = """
        \\begin{table}[htbp]
        \\centering
        $tab1
        \\caption{Parameter estimates for the school full simulated choice model using Monte Carlo methods.}
        \\end{table}
        """

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


        init_params = vcat(zeros(3 + J - 1), 1)
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
        result = optimize(
            p -> loglik_joint_simulated_GH(p, test_scores, sports, distance, y, k),
            lb,
            ub,
            init_params,
            Fminbox(BFGS())
        )

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
        parameter_xi = ["xi_$j" for j in 1:J]  # creates ["xi_1", "xi_2", ..., "xi_J"]

        parameters = vcat(["alpha", "beta1", "beta2"], parameter_xi, ["sigma_b"])
        vals1 = vcat([alpha_hat, beta_hat[1], beta_hat[2]], xi_hat, [sigma_b_hat])

        df1 = DataFrame(
            Parameter=parameters,
            Estimate=round.(vals1, digits=6)
        )

        tab1 = String(latexify(df1, env=:tabular, latex=false))

        latex_table1 = """
        \\begin{table}[htbp]
        \\centering
        $tab1
        \\caption{Parameter estimates for the school full simulated choice model using Gaussian Quadrature methods.}
        \\end{table}
        """

        println(latex_table1)


        ### 10
        println("--------------------------------")
        println("Problem 2.10")
        println("Computing the estimates using methods of moment...")
        println("--------------------------------")



        function msm_moments(params, test_scores, sports, distance, y, R)
            N, J = size(distance)
            alpha = params[1]
            beta1_mu, beta2 = params[2:3]
            xi = [0.0; params[4:3+J-1]]   # xi1 = 0 for normalization
            sigma_b = params[3+J]

            # Simulate random coefficients
            beta1_draws = rand(Normal(beta1_mu, sigma_b), R)

            # Initialize simulated probabilities P_ij
            P = zeros(N, J)

            for i in 1:N
                P_i = zeros(J)
                for r in 1:R
                    beta1 = beta1_draws[r]
                    # compute utility for household i
                    U = [beta1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j] for j in 1:J]
                    expU = exp.(U .- maximum(U))  # numerical stability
                    P_i .+= expU ./ sum(expU)
                end
                P[i, :] .= P_i ./ R   # average over draws
            end

            # Instruments: test_scores, sports, distance
            # Compute moments per household-school-instrument
            g = zeros(N, J * 3)  # each school has 3 instruments
            for i in 1:N
                for j in 1:J
                    z_ij = [test_scores[j], sports[j], distance[i, j]]
                    g[i, (3*(j-1)+1):(3*j)] .= (Int(y[i] == j) - P[i, j]) .* z_ij
                end
            end

            return g


        end



        function msm_objective(params, test_scores, sports, distance, y, R)
            g_matrix = msm_moments(params, test_scores, sports, distance, y, R)
            g_vec = vec(g_matrix)      # flatten into 1D vector
            return dot(g_vec, g_vec)   # now works
        end

        # Initial values
        init_params = vcat(zeros(3 + J - 1), 1)

        # GMM estimation
        R = 100
        lb = fill(-Inf, length(init_params))
        lb[end] = 1e-6         # last parameter > 0

        ub = fill(Inf, length(init_params))  # no upper bounds
        result = optimize(
            p -> msm_objective(p, test_scores, sports, distance, y, R),
            lb,
            ub,
            init_params,
            Fminbox(BFGS())
        )


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
        parameter_xi = ["xi_$j" for j in 1:J]  # creates ["xi_1", "xi_2", ..., "xi_J"]

        parameters = vcat(["alpha", "beta1", "beta2"], parameter_xi, ["sigma_b"])
        vals1 = vcat([alpha_hat, beta_hat[1], beta_hat[2]], xi_hat, [sigma_b_hat])

        df1 = DataFrame(
            Parameter=parameters,
            Estimate=round.(vals1, digits=6)
        )

        tab1 = String(latexify(df1, env=:tabular, latex=false))

        latex_table1 = """
        \\begin{table}[htbp]
        \\centering
        $tab1
        \\caption{Parameter estimates for the school full simulated choice model using Simulated Method of Moments.}
        \\end{table}
        """

        println(latex_table1)

        ### 9
        println("--------------------------------")
        println("Problem 2.9")
        println("Computing the Jacobian of the Moments")
        println("--------------------------------")

        function jacobian_msm(params, test_scores, sports, distance, y, R)
            N, J = size(distance)
            K = length(params)
            L = N * J * 3
            G = zeros(L, K)

            # For each household, school, instrument
            idx = 1
            beta1_mu, beta2 = params[2:3]
            xi = [0.0; params[4:3+J-1]]
            alpha = params[1]
            sigma_b = params[3+J]

            # Simulate random coefficients
            beta1_draws = rand(Normal(beta1_mu, sigma_b), R)

            for i in 1:N
                P_i = zeros(J)
                dU_dparams = zeros(J, K)
                # Compute average choice probabilities and derivatives
                for r in 1:R
                    beta1 = beta1_draws[r]
                    U = [beta1 * test_scores[j] + beta2 * sports[j] + xi[j] - alpha * distance[i, j] for j in 1:J]
                    expU = exp.(U .- maximum(U))
                    P_r = expU ./ sum(expU)
                    P_i .+= P_r

                    # Derivatives of U w.r.t. params
                    for j in 1:J
                        dU = zeros(K)
                        dU[1] = -distance[i, j]             # alpha
                        dU[2] = test_scores[j]             # beta1_mu
                        dU[3] = sports[j]                  # beta2
                        if j > 1
                            dU[3+(j-1)] = 1.0  # xi_2,...xi_J
                        end
                        # sigma_b derivative can be added if desired
                        dU_dparams[j, :] .+= dU
                    end
                end
                P_i ./= R
                dU_dparams ./= R

                # Compute Jacobian entries
                for j in 1:J
                    z_ij = [test_scores[j], sports[j], distance[i, j]]
                    for k_inst in 1:3  # three instruments
                        for m in 1:K
                            G[idx, m] = -z_ij[k_inst] * (dU_dparams[j, m] * P_i[j] - sum(P_i .* dU_dparams[:, m]))
                        end
                        idx += 1
                    end
                end
            end
            return G
        end

        jacobian = jacobian_msm(params_hat, test_scores, sports, distance, y, 100)
        println("Jacobian size: ", size(jacobian))
        println("The Jacobian matrix G is ", jacobian)

        df1 = DataFrame(jacobian)
        tab_jacobian = String(latexify(df1, env=:tabular, latex=false))
        latex_table_jacobian = """
        \\begin{table}[htbp]
        \\centering
        $tab_jacobian
        \\caption{Jacobian matrix of the moments.}
        \\end{table}
        """
        println(latex_table_jacobian)

        ### 11
        println("--------------------------------")
        println("Problem 2.11")
        println("Estimating efficient MSM...")
        println("--------------------------------")

        function msm_objective_efficient(params, test_scores, sports, distance, y, R)
            g_all = msm_moments(params, test_scores, sports, distance, y, R)  # N x L
            N, L = size(g_all)

            # Compute mean of moments across households
            g_mean = mean(g_all, dims=1)             # 1 x L

            # Covariance matrix of moments (L x L)
            Omega = (g_all .- g_mean)' * (g_all .- g_mean) / N
            W = inv(Omega + 1e-8 * I)                  # add small regularization for numerical stability

            # Stack moments across households
            g = vec(sum(g_all, dims=1))              # L x 1

            return dot(g, W * g)
        end

        # Initial values
        init_params = vcat(zeros(3 + J - 1), 1)

        # GMM estimation
        R = 100
        lb = fill(-Inf, length(init_params))
        lb[end] = 1e-6         # last parameter > 0

        ub = fill(Inf, length(init_params))  # no upper bounds
        result = optimize(
            p -> msm_objective_efficient(p, test_scores, sports, distance, y, R),
            lb,
            ub,
            init_params,
            Fminbox(BFGS())
        )


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
        parameter_xi = ["xi_$j" for j in 1:J]  # creates ["xi_1", "xi_2", ..., "xi_J"]

        parameters = vcat(["alpha", "beta1", "beta2"], parameter_xi, ["sigma_b"])
        vals1 = vcat([alpha_hat, beta_hat[1], beta_hat[2]], xi_hat, [sigma_b_hat])

        df1 = DataFrame(
            Parameter=parameters,
            Estimate=round.(vals1, digits=6)
        )

        tab1 = String(latexify(df1, env=:tabular, latex=false))

        latex_table1 = """
        \\begin{table}[htbp]
        \\centering
        $tab1
        \\caption{Parameter estimates for the school full simulated choice model using Simulated Method of Moments.}
        \\end{table}
        """

        println(latex_table1)

    end
end