
# output the obj and coefficients of the trained models 
function model_summarize(model,data_type_list)
    optimal_obj = JuMP.objective_value(model);
    beta_opt = JuMP.value.(model[:beta])
    beta_opt_intercept = JuMP.value.(model[:beta0])
    #time = solve_time(model);
    if data_type_list[1] == "cont"
        beta_opt_cont = JuMP.value.(model[:beta_cont])
        return optimal_obj, beta_opt_cont, beta_opt, beta_opt_intercept
    end
    return optimal_obj, nothing, beta_opt, beta_opt_intercept
end

# calculate misclassification error
function misclassification(X_test, y_test, beta_opt, beta_opt_intercept, data_type_list, beta_opt_cont=nothing, X_cont_test=nothing)
    if data_type_list[1] == "cont"
        p_hat_computed = 1 ./(1 .+ exp.(-(X_test * beta_opt .+ X_cont_test * beta_opt_cont .+ beta_opt_intercept))); #classify according to the betas
    else
        p_hat_computed = 1 ./(1 .+ exp.(-(X_test * beta_opt .+ beta_opt_intercept))); #classify according to the betas
    end
    
    predictions_computed = [x > 0.5 ? 1 : x < 0.5 ? -1 : rand([-1, 1]) for x in p_hat_computed]

    mc = sum(predictions_computed .!= y_test)
    return mc
end

# calculate adaptive calibration error
function calibration_metric_ACE(X_test, y_test, beta_opt, beta_opt_intercept, num_bins, data_type_list, beta_opt_cont=nothing, X_cont_test=nothing)
    N = size(X_test)[1]

    if data_type_list[1] == "cont"
        predicted_risks = 1.0 ./ (1.0 .+ exp.(-(X_cont_test * beta_opt_cont .+ X_test * beta_opt .+ beta_opt_intercept)))
    else        
        predicted_risks = 1.0 ./ (1.0 .+ exp.(-(X_test * beta_opt .+ beta_opt_intercept)))
    end

# Sort the data
    sorted_risks = sort(predicted_risks)
    indices_bins = zeros(Int, num_bins)
# Determine bin size
    bin_size = N ÷ num_bins  # Integer division

    indices_bins[1] = 1
    for i in 2:num_bins
        indices_bins[i] = indices_bins[i-1] + bin_size
    end

    if N - bin_size * (num_bins - 1) > bin_size
        diff = N - bin_size * num_bins
        count = 1
        for i in num_bins - diff + 2 : num_bins
            indices_bins[i] += count
            count += 1
        end
    end

# Create bins
    bins = []
    for i in 1:num_bins
        if i != num_bins
            push!(bins, sorted_risks[indices_bins[i]:indices_bins[i+1]-1])
        else
            push!(bins, sorted_risks[indices_bins[i]:N])
        end
    end

    mean_predicted_risks = Float64[]
    observed_risks = Float64[]
    y_test_01 = replace(y_test, -1 => 0)  # Create new list with -1 replaced by 0


# Calculate mean predicted risk and observed risk for each bin
    for bin in bins
    # Find indices of the elements in the original array
        indices = findall(x -> x in bin, predicted_risks)

        # Calculate mean predicted risk
        mean_predicted_risk = mean(bin)
        push!(mean_predicted_risks, mean_predicted_risk)

        # Calculate observed risk
        observed_risk = mean(y_test_01[indices])
        push!(observed_risks, observed_risk)
    end

# Calculate calibration error for each bin
    CAL = 0
    for i in 1:num_bins
        if i != num_bins
            frac = (indices_bins[i+1] - indices_bins[i]) / N
        else
            frac = (N + 1  - indices_bins[i]) / N
        end
        CAL = CAL + frac * abs(mean_predicted_risks[i] - observed_risks[i])
    end

    return CAL
end

# calculate auc

function calculate_AUC(X_test, y_test, beta_opt, beta_opt_intercept, data_type_list, beta_opt_cont=nothing, X_cont_test=nothing)
    if data_type_list[1] == "cont"
        scores = X_cont_test * beta_opt_cont .+ X_test * beta_opt .+ beta_opt_intercept
    else
        scores = X_test * beta_opt .+ beta_opt_intercept
    end

    positives = findall(x -> x == +1, y_test)
    negatives = findall(x -> x == -1, y_test)

    n_pos = length(positives)
    n_neg = length(negatives)

    if n_pos == 0 || n_neg == 0
        return NaN  # Handling cases with no positive or negative samples
    end

    auc = 0
    for i in positives
        for k in negatives
            auc += scores[i] > scores[k]
        end
    end

    return auc / (n_pos * n_neg)
end

