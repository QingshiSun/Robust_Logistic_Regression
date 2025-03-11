#add the exp-cone constraints
function softplus(model, t, linear_transform) 
    z = @variable(model, [1:2])
    @constraint(model, sum(z) <= 1.0)
    @constraint(model, [linear_transform - t, 1, z[1]] in MOI.ExponentialCone())
    @constraint(model, [-t, 1, z[2]] in MOI.ExponentialCone())
end



function build_logit_model_cate(X, y, groups, regular, regular_coef, data_type_list, X_cont=nothing)
    N, n = size(X)
    model = Model() 
    rel_gap = 10^-7
    set_optimizer(model, MosekTools.Optimizer)
    set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1) 
    set_optimizer_attributes(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => rel_gap)
    set_optimizer_attributes(model, "MSK_DPAR_INTPNT_TOL_REL_GAP" => rel_gap)
    @variable(model, beta[1:n]) 
    @variable(model, beta0)
    @variable(model, t[1:N]) 

    if data_type_list[1] == "cont"
        n_cont = size(X_cont)[2]
        @variable(model, beta_cont[1:n_cont])
    end 

    if data_type_list[1] == "cont"
        for i in 1:N
            u = -(X_cont[i,:]'* beta_cont + X[i, :]' * beta + beta0) * y[i]
            softplus(model, t[i], u)
        end
    else
        for i in 1:N
            u = -(X[i, :]' * beta + beta0) * y[i]
            softplus(model, t[i], u)
        end
    end

    for g in groups
        if length(g) >= 3
            @constraint(model, beta[g[end]] == 0) 
        end
    end

# set the regularization
    if data_type_list[1] == "cont"
        if regular == 2 # ridge
            @variable(model, 0.0 <= reg)
            @constraint(model, [reg; [beta_cont; beta; beta0]] in SecondOrderCone())
            @objective(model, Min, sum(t)/N + (regular_coef * reg))
        elseif regular == 1 #lasso
            @variable(model, 0.0 <= reg)
            @constraint(model, [reg; [beta_cont; beta; beta0]] in MOI.NormOneCone(n_cont + n + 2))
            @objective(model, Min, sum(t)/N + (regular_coef * reg))
        else #no regularization
            @objective(model, Min, sum(t)/N)
        end
    else
        if regular == 2 
            @variable(model, 0.0 <= reg)
            @constraint(model, [reg; [beta; beta0]] in SecondOrderCone())
            @objective(model, Min, sum(t)/N + (regular_coef * reg))
        elseif regular == 1 
            @variable(model, 0.0 <= reg)
            @constraint(model, [reg; [beta; beta0]] in MOI.NormOneCone(n + 2))
            @objective(model, Min, sum(t)/N + (regular_coef * reg))
        else 
            @objective(model, Min, sum(t)/N)
        end
    end
    return model
end


function logistic_regression_cate(X, y, groups, regular, regular_coef, data_type_list, X_cont=nothing)
    model = build_logit_model_cate(X, y, groups, regular, regular_coef, data_type_list, X_cont)
    set_silent(model)
    JuMP.optimize!(model)
    if termination_status(model) != OPTIMAL 
        error("Solution is not optimal.")
    end
    solver_time = solve_time(model);
    return model, solver_time
end


# cross validation for regularization cofficient selection 
function cv_wasserstein_cate(X, y, groups, regular_coef, regular, metric_type, num_bins, data_type_list, X_cont=nothing)
    N = size(X)[1]
    break_points = round.(Int,LinRange(1,N,5+1)) 
    vsets = [s:e-(e<N)*1 for (s,e) in zip(break_points[1:end-1],break_points[2:end])] # indices of data points in each fold

    errors = zeros(5)
    for i = 1:5 
        X_valid = X[vsets[i], :]
        y_valid = y[vsets[i]] 

        X_train = X[Not(vsets[i]),:] #train set is all, except for i-th fold
        y_train = y[Not(vsets[i])]


        X_cont_valid = nothing
        X_cont_train = nothing
        if data_type_list[1] == "cont"
            X_cont_valid = X_cont[vsets[i], :] #validation set is the i-th fold
            X_cont_train = X_cont[Not(vsets[i]),:] 
        end

        model, solver_time = logistic_regression_cate(X_train, y_train, groups, regular, regular_coef, data_type_list, X_cont_train)
    
        optimal_obj, beta_opt_cont, beta_opt, beta_opt_intercept = model_summarize(model,data_type_list)

        if metric_type == "ACE"
            test_error = calibration_metric_ACE(X_valid, y_valid, beta_opt, beta_opt_intercept, num_bins, data_type_list, beta_opt_cont, X_cont_valid)
        else
            test_error = calculate_AUC(X_valid, y_valid, beta_opt, beta_opt_intercept, data_type_list, beta_opt_cont, X_cont_valid)
        end

        push!(errors, test_error)

    end
    avg_cv_error = mean(errors)
    return avg_cv_error
end


function cv_wasserstein_DRO_cate(X, y, groups, rounded_cate_gammas, regular_coef, regular, radius, metric_type, num_bins, data_type_list, b_unbounded_vector = nothing, X_cont=nothing)
    #inputs:
#X, y: categorical covariates and labels in the training dataset
# groups: for each categorical feature, its range of encoded features after one hot encoding
# rounded_cate_gammas: the weight parameters of categorical features
# epsilon: ambiguity set radius
#dual: whether we solve models by dual cone 
#regular_coef, regular: used for regualarized logistic regression. corresponding to the case where the radius is 0 (no perturbation allowed)
# metric_type: the metric used to evaluate the model performace (calibration error or AUC)
# radius: ambibuity set radius
# num_bins: number of bins used in calibration error calculation
#data_type_list: the types of data in the dataset
# laplace_b_unbounded_vector: the scale parameter of the lapalce distribution for each numerical feature
# X_cont_train: the numerical feature part of the training dataset
    
    N = size(X)[1]
    break_points = round.(Int,LinRange(1,N,5+1)) 
    vsets = [s:e-(e<N)*1 for (s,e) in zip(break_points[1:end-1],break_points[2:end])] #  indices of data points in each fold

    errors = zeros(5)
    for i = 1:5 
        X_valid = X[vsets[i], :]
        y_valid = y[vsets[i]] 

        X_train = X[Not(vsets[i]),:] #train set is all, except for i-th fold
        y_train = y[Not(vsets[i])]


        X_cont_valid = nothing
        X_cont_train = nothing
        if data_type_list[1] == "cont"
            X_cont_valid = X_cont[vsets[i], :] #validation set is the i-th fold
            X_cont_train = X_cont[Not(vsets[i]),:] 
        end

        if radius == 0
            model_DRO_DP, solver_time = logistic_regression_cate(X_train, y_train, groups, regular, regular_coef, data_type_list, X_cont_train)
        else
            model_DRO_DP, solver_time_DRO_DP, build_time_DRO_DP, build_byte_DRO_DP = DP_wasserstein_cate_oneset(X_train, y_train, groups, rounded_cate_gammas, radius, 1, data_type_list, b_unbounded_vector, X_cont_train)
        end


        optimal_obj_DRO_DP, beta_opt_cont_DRO_DP, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP = model_summarize(model_DRO_DP, data_type_list)
        if metric_type == "ACE"
            test_error = calibration_metric_ACE(X_valid, y_valid, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP, num_bins, data_type_list, beta_opt_cont_DRO_DP, X_cont_valid)
        else
            test_error = calculate_AUC(X_valid, y_valid, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP, data_type_list, beta_opt_cont_DRO_DP, X_cont_valid)
        end
        push!(errors, test_error)

    end
    avg_cv_error = mean(errors)
    return avg_cv_error
end
