using JuMP 
using LinearAlgebra 
import Random 
import MosekTools 
import MathOptInterface 
const MOI = MathOptInterface 
using Dualization
using DelimitedFiles 
using InvertedIndices
using JLD2 
using Random, Distributions
using DataFrames
using DataStructures
using GLPK
using Base.Iterators: product
using Serialization
using Statistics
using Plots
using StatsPlots  
using BenchmarkTools
using CSV
using Dates

include("perturbations.jl")
include("cutting.jl")
include("graph.jl")
include("metrics.jl")
include("LR.jl")


dataset_list = ["credit"]
high_seed = 12
prob_mean_list = [0.6, 0.7, 0.8, 0.9]

prob_std = 0.2


local_path = ""
lambda_list = [0.5, 0.65, 0.75, 0.8, 0.85, 0.9, 0.93, 0.96, 0.99]


kappa = 100000
train_fraction = 0.8
certainty_interval_probability = 0.4
gamma_precision_list = [-1, 0, 1]

regular_coef_list = [0, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000]
data_type_list = ["nothing", "nothing", "cate"]
set_range = [1]
regular_type = 1
metric_type = "ACE"
maximum_iteration = 6000 
output_length = 189


# decide how to change the original sampled probabilities. In this version, only expected and 7 unexpected cases are tested
unexpected_level_list = [0, -0.2, -0.1, 0.1, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2]

exception_results = DataFrame(dataset=String[], prob=Float64[], GammaDigit=Int[], radius=Float64[], method=String[], model_status=[])

df = DataFrame()
data_type = data_type_list[1] * "_" * data_type_list[2] * "_" * data_type_list[3]


for dataset_name in dataset_list

    #import data and shuffle
    data_combined = readdlm(local_path * dataset_name * "-cooked.csv", ',', Float64) 
    Random.seed!(high_seed)
    shuffled_indices = Random.shuffle(1:size(data_combined, 1))

    if data_type_list[1] == "cont"
        X_cont = readdlm(local_path * dataset_name * "-cont-cooked.csv", ',', Float64)
        if size(X_cont)[1] != size(data_combined)[1]
            error("different sizes in numerical and categorical data points")
        end
        X_cont = X_cont[shuffled_indices, :] 
    end
    data_combined = data_combined[shuffled_indices, :] 
    Random.seed!()

    X = data_combined[:,1:end-1]
    y = data_combined[:,end]

    train_size = floor(Int, size(data_combined)[1] * train_fraction)
    test_size = size(data_combined)[1] - train_size

    X_cate_train = data_combined[1:train_size,1:end-1]
    X_cate_test = data_combined[(train_size + 1):end, 1:end-1]

    y_train = data_combined[1:train_size,end]
    y_test = data_combined[(train_size + 1):end,end]

    num_train_data, num_encoded_cate_features = size(X_cate_train)
    num_test_data = size(X_cate_test)[1]

    # process continuous data
    num_cont_features = 0
    X_cont_train = nothing
    X_cont_test = nothing 
    if data_type_list[1] == "cont"
        num_cont_features = size(X_cont)[2]
        X_cont_train = X_cont[1:train_size,:]
        X_cont_test = X_cont[(train_size + 1):end, :]
    end
        
    #groups 
    groups = vec(readdlm(local_path * dataset_name * "-cooked-key.csv", ',', String))[1:end-1] 
    groups = [eval(Meta.parse(replace(g, r"-" => ":"))) for g in groups]
    T = length(groups)

    # LR
    model_LR, solver_times_LR = logistic_regression_cate(X_cate_train, y_train, groups, 0, 0, data_type_list, X_cont_train)
    optimal_obj_LR, beta_opt_cont_LR, beta_opt_cate_LR, beta_opt_intercept_LR = model_summarize(model_LR, data_type_list)
    # store the LR coefficients
    serialize(local_path * "model_LR_obj_coef_" * dataset_name * "_" * data_type * "_shuffle_seed_$(high_seed)" * ".bin", (optimal_obj_LR, beta_opt_cont_LR, beta_opt_cate_LR, beta_opt_intercept_LR))
    println("LR: dataset " * dataset_name * " = ",  [optimal_obj_LR, beta_opt_cont_LR, beta_opt_cate_LR, beta_opt_intercept_LR])

    # RLR
    avg_cv_errors_RLR = Float64[]
    bin_num = 10
    # cv for lasso coefficients
    for regular_coef in regular_coef_list
        println(regular_coef)
        avg_cv_error = cv_wasserstein_cate(X_cate_train, y_train, groups, regular_coef, regular_type, metric_type, bin_num, data_type_list, X_cont_train) # 10 bins; CV for CCG algorithm
        push!(avg_cv_errors_RLR, avg_cv_error)
    end
    
    if metric_type == "ACE"
        regular_coef_opt_RLR = regular_coef_list[argmin(avg_cv_errors_RLR)]
    else
        regular_coef_opt_RLR = regular_coef_list[argmax(avg_cv_errors_RLR)]
    end

    model_RLR, solver_times_RLR = logistic_regression_cate(X_cate_train, y_train, groups, regular_type, regular_coef_opt_RLR, data_type_list, X_cont_train)
    optimal_obj_RLR, beta_opt_cont_RLR, beta_opt_cate_RLR, beta_opt_intercept_RLR = model_summarize(model_RLR, data_type_list)
    serialize(local_path * "model_RLR_obj_coef_" * dataset_name * "_" * data_type * "_shuffle_seed_$(high_seed)_" * "regular_type_$(regular_type)" * ".bin", (optimal_obj_RLR, beta_opt_cont_RLR, beta_opt_cate_RLR, beta_opt_intercept_RLR, regular_coef_opt_RLR))
    println("RLR: dataset " * dataset_name * " R type $(regular_type) R coeff $(regular_coef_opt_RLR)= ",  [optimal_obj_RLR, beta_opt_cont_RLR, beta_opt_cate_RLR, beta_opt_intercept_RLR])


    for prob_mean in prob_mean_list

        # generate prob of certainty; 
        probability_seed_cate = Int(prob_mean*1000 + num_train_data + num_encoded_cate_features + num_cont_features)
        Random.seed!(probability_seed_cate)
        probability_certainty_cate_matrix = generate_probability_certainty_cate(num_train_data, groups, prob_mean, prob_std)
        probability_certainty_cont_vector = nothing
        if data_type_list[1] == "cont"
            probability_certainty_cont_vector = generate_probability_certainty_cont(num_cont_features, prob_mean, prob_std)
            println("prob: dataset " * dataset_name * " probability $(prob_mean)_cont = ",  probability_certainty_cont_vector)  
        end
        Random.seed!()

        #generate  weights  for categorical features
        cate_gammas = generate_cate_gammas(probability_certainty_cate_matrix, groups)
        probability_certainty_cate_matrix_test = Array{Float64}(undef, num_test_data, T)
        for i in 1:num_test_data
            probability_certainty_cate_matrix_test[i,:] = probability_certainty_cate_matrix[1,:]
        end
        serialize(local_path * "sampled_prob_mean_$(prob_mean)_prob_std_$(prob_std)_" * dataset_name * "_"  * data_type * "_shuffle_seed_$(high_seed)_prob_seed_$(probability_seed_cate)_cate" * ".bin", (probability_certainty_cate_matrix))
        println("prob: dataset " * dataset_name * " probability $(prob_mean)_cate = ",  probability_certainty_cate_matrix[1,:])  

        #generate  weights  for numerical features
        laplace_b_unbounded_vector = nothing
        if data_type_list[1] == "cont"
            certainty_bounds = generate_certainty_interval_bound_normal(X_cont_train, certainty_interval_probability)
            laplace_b_unbounded_vector = calculate_laplace_b_unbounded(probability_certainty_cont_vector, certainty_bounds)
            serialize(local_path * "sampled_prob_mean_$(prob_mean)_prob_std_$(prob_std)_" * dataset_name * "_"  * data_type * "_shuffle_seed_$(high_seed)_prob_seed_$(probability_seed_cate)_cont" * ".bin", (laplace_b_unbounded_vector))
        end

        for digit in gamma_precision_list
            # round weights (gamma) to improve calculation efficiency
            if digit == -1
                rounded_cate_gammas = ones(Float64, 1, T)
                final_laplace_b_unbounded_vector = ones(Float64, 1, num_cont_features)
            else
                rounded_cate_gammas = round.(cate_gammas, digits=digit)
                final_laplace_b_unbounded_vector = laplace_b_unbounded_vector
            end
            serialize(local_path * "rounded_gammas_digit_$(digit)_prob_mean_$(prob_mean)_prob_std_$(prob_std)_" * dataset_name * "_" * data_type * "_shuffle_seed_$(high_seed)_prob_seed_$(probability_seed_cate)" * ".bin", (rounded_cate_gammas))
            println("gamma: dataset " * dataset_name * " probability $(prob_mean) = ",  rounded_cate_gammas)
            println("b: dataset " * dataset_name * " probability $(prob_mean)_cont = ",  final_laplace_b_unbounded_vector)  

            # given epsilon
            epsilon_lists = log.(lambda_list).*(-1)
            for epsilon in epsilon_lists
                lambda_epsilon = exp(- epsilon /1)

                # specify the output format
                vec_dict = Dict("LR" => [], "RLR" => [], "DP" => [])
                for method in keys(vec_dict)
                    push!(vec_dict[method],dataset_name)
                    push!(vec_dict[method], data_type)
                    push!(vec_dict[method], method)
                    push!(vec_dict[method], lambda_epsilon)
                    push!(vec_dict[method], prob_mean)
                    push!(vec_dict[method], prob_std)
                    push!(vec_dict[method], digit)
                    push!(vec_dict[method], probability_seed_cate)
                    if method == "RLR"
                        push!(vec_dict[method], regular_coef_opt_RLR)
                        push!(vec_dict[method], regular_type)
                    else
                        push!(vec_dict[method], NaN)
                        push!(vec_dict[method], NaN)
                    end
                end

                push!(vec_dict["LR"], optimal_obj_LR)
                push!(vec_dict["RLR"], optimal_obj_RLR)

                for standard_method in ["LR", "RLR"]
                    for count in 12:18
                        push!(vec_dict[standard_method], NaN)
                    end
                end


                # train DRO model
                DRO_DP_results, DRO_DP_time, DRO_DP_byte, DRO_DP_gctime, DRO_DP_memalloc  = @timed DP_wasserstein_cate_oneset(X_cate_train, y_train, groups, rounded_cate_gammas, epsilon, 1, data_type_list, final_laplace_b_unbounded_vector, X_cont_train)
                                
                model_DRO_DP, solver_time_DRO_DP, build_time_DRO_DP, build_byte_DRO_DP = DRO_DP_results
                if  (termination_status(model_DRO_DP) == OPTIMAL)
                    optimal_obj_DRO_DP, beta_opt_cont_DRO_DP, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP = model_summarize(model_DRO_DP, data_type_list)
                    serialize(local_path * "model_DRO_DP_" * dataset_name * "_" * data_type * "_shuffle_seed_$(high_seed)_prob_seed_$(probability_seed_cate)_digit_$(digit)_prob_mean_$(prob_mean)_prob_std_$(prob_std)_lambda_$(lambda_epsilon)_epsilon_$(epsilon)_obj_coef" * ".bin", (optimal_obj_DRO_DP, beta_opt_cont_DRO_DP, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP))
                    println("DRO DP: dataset " * dataset_name * " probability $(prob_mean) lambda $(lambda_epsilon) = ",  [optimal_obj_DRO_DP, beta_opt_cont_DRO_DP, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP])
                    push!(vec_dict["DP"], optimal_obj_DRO_DP)
                    push!(vec_dict["DP"], DRO_DP_time)
                    push!(vec_dict["DP"], DRO_DP_byte)
                    push!(vec_dict["DP"], solver_time_DRO_DP)
                    push!(vec_dict["DP"], build_time_DRO_DP)
                    push!(vec_dict["DP"], NaN)
                    push!(vec_dict["DP"], NaN)
                    push!(vec_dict["DP"], NaN)
                else
                    serialize(local_path * "model_DRO_DP_" * dataset_name * "_" * data_type * "_shuffle_seed_$(high_seed)_prob_seed_$(probability_seed_cate)_digit_$(digit)_prob_mean_$(prob_mean)_prob_std_$(prob_std)_lambda_$(lambda_epsilon)_epsilon_$(epsilon)_obj_coef" * ".bin", (termination_status(model_DRO_DP)))
                    println("DRO DP: dataset " * dataset_name * " probability $(prob_mean) lambda $(lambda_epsilon) = ",  termination_status(model_DRO_DP))
                    push!(exception_results, (dataset_name, prob_mean, digit, lambda_epsilon, "DP", termination_status(model_DRO_DP)))
                    CSV.write(local_path * dataset_name * "_" * data_type * "_output_dp_exceptions.csv", exception_results)
                    continue
                end
                

                # performance on non perturbed test data
                non_pertubed_calibration_ACE_LR = calibration_metric_ACE(X_cate_test, y_test, beta_opt_cate_LR, beta_opt_intercept_LR, 10, data_type_list, beta_opt_cont_LR, X_cont_test)
                push!(vec_dict["LR"], non_pertubed_calibration_ACE_LR)
                non_pertubed_AUC_LR= calculate_AUC(X_cate_test, y_test, beta_opt_cate_LR, beta_opt_intercept_LR, data_type_list, beta_opt_cont_LR, X_cont_test)
                push!(vec_dict["LR"], non_pertubed_AUC_LR)
                non_pertubed_error_LR = misclassification(X_cate_test, y_test, beta_opt_cate_LR, beta_opt_intercept_LR, data_type_list, beta_opt_cont_LR, X_cont_test)/test_size
                push!(vec_dict["LR"], non_pertubed_error_LR)

                non_pertubed_calibration_ACE_RLR = calibration_metric_ACE(X_cate_test, y_test, beta_opt_cate_RLR, beta_opt_intercept_RLR, 10, data_type_list, beta_opt_cont_RLR, X_cont_test)
                push!(vec_dict["RLR"], non_pertubed_calibration_ACE_RLR)
                non_pertubed_AUC_RLR= calculate_AUC(X_cate_test, y_test, beta_opt_cate_RLR, beta_opt_intercept_RLR, data_type_list, beta_opt_cont_RLR, X_cont_test)
                push!(vec_dict["RLR"], non_pertubed_AUC_RLR)
                non_pertubed_error_RLR = misclassification(X_cate_test, y_test, beta_opt_cate_RLR, beta_opt_intercept_RLR, data_type_list, beta_opt_cont_RLR, X_cont_test)/test_size
                push!(vec_dict["RLR"], non_pertubed_error_RLR)

                non_pertubed_calibration_ACE_DRO_DP = calibration_metric_ACE(X_cate_test, y_test, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP, 10, data_type_list, beta_opt_cont_DRO_DP, X_cont_test)
                push!(vec_dict["DP"], non_pertubed_calibration_ACE_DRO_DP)
                non_pertubed_AUC_DRO_DP = calculate_AUC(X_cate_test, y_test, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP, data_type_list, beta_opt_cont_DRO_DP, X_cont_test)
                push!(vec_dict["DP"], non_pertubed_AUC_DRO_DP)
                non_pertubed_error_DRO_DP = misclassification(X_cate_test, y_test, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP, data_type_list, beta_opt_cont_DRO_DP, X_cont_test)/test_size
                push!(vec_dict["DP"], non_pertubed_error_DRO_DP)


                # when level id = 1, get prob for expected perturbation; for other levels tested before, get prob for unexpected perturbation
                for (level_id, level) in enumerate(unexpected_level_list)
                    if level_id <= 4 
                        probability_certainty_cate_matrix_test_revised = deepcopy(probability_certainty_cate_matrix_test) .+ level
                        for i in 1:num_test_data
                            for (group_id, group) in enumerate(groups)
                                probability_certainty_cate_matrix_test_revised[i, group_id] = min(0.9999, probability_certainty_cate_matrix_test_revised[i, group_id])
                                if length(group) == 1
                                    probability_certainty_cate_matrix_test_revised[i, group_id] = max(1/2, probability_certainty_cate_matrix_test_revised[i, group_id])
                                else
                                    probability_certainty_cate_matrix_test_revised[i, group_id] = max(1/length(group), probability_certainty_cate_matrix_test_revised[i, group_id])
                                end
                            end
                        end
                        serialize(local_path * "unexpected_prob_level_$(level)_prob_mean_$(prob_mean)_prob_std_$(prob_std)_" * dataset_name * "_" * data_type * "_shuffle_seed_$(high_seed)_prob_seed_$(probability_seed_cate)_cate" * ".bin", (probability_certainty_cate_matrix_test_revised[1,:]))

                        probability_certainty_cont_vector_test_revised = nothing
                        if data_type_list[1] == "cont"
                            probability_certainty_cont_vector_test_revised = deepcopy(probability_certainty_cont_vector) .+ level
                            for feature_id in 1:num_cont_features
                                probability_certainty_cont_vector_test_revised[feature_id] = min(0.9999, probability_certainty_cont_vector_test_revised[feature_id])
                                probability_certainty_cont_vector_test_revised[feature_id] = max(0.0001, probability_certainty_cont_vector_test_revised[feature_id])
                            end
                        end
                        serialize(local_path * "unexpected_prob_level_$(level)_prob_mean_$(prob_mean)_prob_std_$(prob_std)_" * dataset_name * "_" * data_type * "_shuffle_seed_$(high_seed)_prob_seed_$(probability_seed_cate)_cont" * ".bin", (probability_certainty_cont_vector_test_revised))
                    elseif (level_id > 8) && (level_id <=12) #
                        Random.seed!(probability_seed_cate + (level_id-1) * 100)
                        probability_certainty_cate_matrix_test_revised_1_row = deepcopy(probability_certainty_cate_matrix_test[1,:])
                        for (group_id, group) in enumerate(groups)
                            upper_bound = min(0.9999, probability_certainty_cate_matrix_test_revised_1_row[group_id] + level)
                            if length(group) == 1
                                lower_bound = max(1/2, probability_certainty_cate_matrix_test_revised_1_row[group_id] - level)
                            else
                                lower_bound = max(1/length(group), probability_certainty_cate_matrix_test_revised_1_row[group_id] - level)
                            end
                            probability_certainty_cate_matrix_test_revised_1_row[group_id] = rand(Uniform(lower_bound, upper_bound))
                        end
                        probability_certainty_cate_matrix_test_revised = Array{Float64}(undef, num_test_data, T)
                        for i in 1:num_test_data
                            probability_certainty_cate_matrix_test_revised[i,:] = probability_certainty_cate_matrix_test_revised_1_row
                        end
                        serialize(local_path * "unexpected_prob_level_$(level)_d_prob_mean_$(prob_mean)_prob_std_$(prob_std)_" * dataset_name * "_" * data_type * "_shuffle_seed_$(high_seed)_prob_seed_$(probability_seed_cate)_cate" * ".bin", (probability_certainty_cate_matrix_test_revised[1,:]))

                        probability_certainty_cont_vector_test_revised = nothing
                        if data_type_list[1] == "cont"
                            probability_certainty_cont_vector_test_revised = Array{Float64}(undef, 1, num_cont_features)
                            for feature_id in 1:num_cont_features
                                upper_bound = min(0.9999, probability_certainty_cont_vector[feature_id] + level)
                                lower_bound = max(0.0001, probability_certainty_cont_vector[feature_id] - level)
                                probability_certainty_cont_vector_test_revised[feature_id] = rand(Uniform(lower_bound, upper_bound))
                            end
                        end
                        serialize(local_path * "unexpected_prob_level_$(level)_s_prob_mean_$(prob_mean)_prob_std_$(prob_std)_" * dataset_name * "_" * data_type * "_shuffle_seed_$(high_seed)_prob_seed_$(probability_seed_cate)_cont" * ".bin", (probability_certainty_cont_vector_test_revised))
                        Random.seed!()
                    else
                        continue
                    end
                    

                    error_LR_list = Float64[]
                    error_DRO_DP_list = Float64[]
                    error_RLR_list = Float64[]
                    
                    calibration_error_ACE_LR_list = Float64[]
                    calibration_error_ACE_DRO_DP_list = Float64[]
                    calibration_error_ACE_RLR_list = Float64[]

                    AUC_LR_list = Float64[]
                    AUC_DRO_DP_list = Float64[]
                    AUC_RLR_list = Float64[]

                    laplace_b_unbounded_vector_perturbed = nothing
                    if data_type_list[1] == "cont"
                        println("probability_certainty_cont_vector_test_revised",probability_certainty_cont_vector_test_revised)
                        laplace_b_unbounded_vector_perturbed = calculate_laplace_b_unbounded(probability_certainty_cont_vector_test_revised, certainty_bounds)
                        println("laplace_b_unbounded_vector_perturbed", laplace_b_unbounded_vector_perturbed)
                    end

                    # simulate perturbations under given prob in 5000 different ways
                    for i in 1:5000
                        
                        Random.seed!(probability_seed_cate + i * 1000 + (level_id-1) * 100)
                        X_cate_test_perturbed = generate_perturbed_cate_features(probability_certainty_cate_matrix_test_revised, groups, X_cate_test)
                        X_cont_test_perturbed = nothing
                        if data_type_list[1] == "cont"
                            X_cont_test_perturbed = X_cont_test + generate_laplace_perturbation(test_size, laplace_b_unbounded_vector_perturbed)
                        end
                        Random.seed!()

                        error_LR = misclassification(X_cate_test_perturbed, y_test, beta_opt_cate_LR, beta_opt_intercept_LR, data_type_list, beta_opt_cont_LR, X_cont_test_perturbed)/test_size
                        push!(error_LR_list, error_LR)
                        calibration_ACE_LR= calibration_metric_ACE(X_cate_test_perturbed, y_test, beta_opt_cate_LR, beta_opt_intercept_LR, 10, data_type_list, beta_opt_cont_LR, X_cont_test_perturbed)
                        push!(calibration_error_ACE_LR_list, calibration_ACE_LR)
                        AUC_LR= calculate_AUC(X_cate_test_perturbed, y_test, beta_opt_cate_LR, beta_opt_intercept_LR, data_type_list, beta_opt_cont_LR, X_cont_test_perturbed)
                        push!(AUC_LR_list, AUC_LR)

                        error_DRO_DP = misclassification(X_cate_test_perturbed, y_test, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP, data_type_list, beta_opt_cont_DRO_DP, X_cont_test_perturbed)/test_size
                        push!(error_DRO_DP_list, error_DRO_DP)
                        calibration_ACE_DRO_DP = calibration_metric_ACE(X_cate_test_perturbed, y_test, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP, 10, data_type_list, beta_opt_cont_DRO_DP, X_cont_test_perturbed)
                        push!(calibration_error_ACE_DRO_DP_list, calibration_ACE_DRO_DP)
                        AUC_DRO_DP= calculate_AUC(X_cate_test_perturbed, y_test, beta_opt_cate_DRO_DP, beta_opt_intercept_DRO_DP, data_type_list, beta_opt_cont_DRO_DP, X_cont_test_perturbed)
                        push!(AUC_DRO_DP_list, AUC_DRO_DP)


                        error_RLR = misclassification(X_cate_test_perturbed, y_test, beta_opt_cate_RLR, beta_opt_intercept_RLR, data_type_list, beta_opt_cont_RLR, X_cont_test_perturbed)/test_size
                        push!(error_RLR_list, error_RLR)
                        calibration_ACE_RLR = calibration_metric_ACE(X_cate_test_perturbed, y_test, beta_opt_cate_RLR, beta_opt_intercept_RLR, 10, data_type_list, beta_opt_cont_RLR, X_cont_test_perturbed)
                        push!(calibration_error_ACE_RLR_list, calibration_ACE_RLR)
                        AUC_RLR= calculate_AUC(X_cate_test_perturbed, y_test, beta_opt_cate_RLR, beta_opt_intercept_RLR, data_type_list, beta_opt_cont_RLR, X_cont_test_perturbed)
                        push!(AUC_RLR_list, AUC_RLR)
                    end

        
                    # calibration
                    worst_calibration_error_ACE_LR = maximum(calibration_error_ACE_LR_list)
                    worst_calibration_error_ACE_RLR = maximum(calibration_error_ACE_RLR_list)
                    worst_calibration_error_ACE_DRO_DP = maximum(calibration_error_ACE_DRO_DP_list)
                    push!(vec_dict["LR"], worst_calibration_error_ACE_LR)
                    push!(vec_dict["RLR"], worst_calibration_error_ACE_RLR)
                    push!(vec_dict["DP"], worst_calibration_error_ACE_DRO_DP)

                    quantile_25_calibration_error_ACE_LR = quantile(calibration_error_ACE_LR_list, 0.25)
                    quantile_25_calibration_error_ACE_RLR = quantile(calibration_error_ACE_RLR_list, 0.25)
                    quantile_25_calibration_error_ACE_DRO_DP = quantile(calibration_error_ACE_DRO_DP_list, 0.25)
                    push!(vec_dict["LR"], quantile_25_calibration_error_ACE_LR)
                    push!(vec_dict["RLR"], quantile_25_calibration_error_ACE_RLR)
                    push!(vec_dict["DP"], quantile_25_calibration_error_ACE_DRO_DP)

                    quantile_50_calibration_error_ACE_LR = quantile(calibration_error_ACE_LR_list, 0.50)
                    quantile_50_calibration_error_ACE_RLR = quantile(calibration_error_ACE_RLR_list, 0.50)
                    quantile_50_calibration_error_ACE_DRO_DP = quantile(calibration_error_ACE_DRO_DP_list, 0.50)
                    push!(vec_dict["LR"], quantile_50_calibration_error_ACE_LR)
                    push!(vec_dict["RLR"], quantile_50_calibration_error_ACE_RLR)
                    push!(vec_dict["DP"], quantile_50_calibration_error_ACE_DRO_DP)

                    quantile_75_calibration_error_ACE_LR = quantile(calibration_error_ACE_LR_list, 0.75)
                    quantile_75_calibration_error_ACE_RLR = quantile(calibration_error_ACE_RLR_list, 0.75)
                    quantile_75_calibration_error_ACE_DRO_DP = quantile(calibration_error_ACE_DRO_DP_list, 0.75)
                    push!(vec_dict["LR"], quantile_75_calibration_error_ACE_LR)
                    push!(vec_dict["RLR"], quantile_75_calibration_error_ACE_RLR)
                    push!(vec_dict["DP"], quantile_75_calibration_error_ACE_DRO_DP)

                    best_calibration_error_ACE_LR = minimum(calibration_error_ACE_LR_list)
                    best_calibration_error_ACE_RLR = minimum(calibration_error_ACE_RLR_list)
                    best_calibration_error_ACE_DRO_DP = minimum(calibration_error_ACE_DRO_DP_list)
                    push!(vec_dict["LR"], best_calibration_error_ACE_LR)
                    push!(vec_dict["RLR"], best_calibration_error_ACE_RLR)
                    push!(vec_dict["DP"], best_calibration_error_ACE_DRO_DP)

                    avg_calibration_error_ACE_LR = mean(calibration_error_ACE_LR_list)
                    avg_calibration_error_ACE_RLR = mean(calibration_error_ACE_RLR_list)
                    avg_calibration_error_ACE_DRO_DP = mean(calibration_error_ACE_DRO_DP_list)
                    push!(vec_dict["LR"], avg_calibration_error_ACE_LR)
                    push!(vec_dict["RLR"], avg_calibration_error_ACE_RLR)
                    push!(vec_dict["DP"], avg_calibration_error_ACE_DRO_DP)


                    std_calibration_error_ACE_LR = std(calibration_error_ACE_LR_list)
                    std_calibration_error_ACE_RLR = std(calibration_error_ACE_RLR_list)
                    std_calibration_error_ACE_DRO_DP = std(calibration_error_ACE_DRO_DP_list)
                    push!(vec_dict["LR"], std_calibration_error_ACE_LR)
                    push!(vec_dict["RLR"], std_calibration_error_ACE_RLR)
                    push!(vec_dict["DP"], std_calibration_error_ACE_DRO_DP)


                    # AUC
                    worst_AUC_LR = minimum(AUC_LR_list)
                    worst_AUC_RLR = minimum(AUC_RLR_list)
                    worst_AUC_DRO_DP = minimum(AUC_DRO_DP_list)
                    push!(vec_dict["LR"], worst_AUC_LR)
                    push!(vec_dict["RLR"], worst_AUC_RLR)
                    push!(vec_dict["DP"], worst_AUC_DRO_DP)


                    quantile_25_AUC_LR = quantile(AUC_LR_list, 0.25)
                    quantile_25_AUC_RLR = quantile(AUC_RLR_list, 0.25)
                    quantile_25_AUC_DRO_DP = quantile(AUC_DRO_DP_list, 0.25)
                    push!(vec_dict["LR"], quantile_25_AUC_LR)
                    push!(vec_dict["RLR"], quantile_25_AUC_RLR)
                    push!(vec_dict["DP"], quantile_25_AUC_DRO_DP)


                    quantile_50_AUC_LR = quantile(AUC_LR_list, 0.50)
                    quantile_50_AUC_RLR = quantile(AUC_RLR_list, 0.50)
                    quantile_50_AUC_DRO_DP = quantile(AUC_DRO_DP_list, 0.50)
                    push!(vec_dict["LR"], quantile_50_AUC_LR)
                    push!(vec_dict["RLR"], quantile_50_AUC_RLR)
                    push!(vec_dict["DP"], quantile_50_AUC_DRO_DP)

                                    
                    quantile_75_AUC_LR = quantile(AUC_LR_list, 0.75)
                    quantile_75_AUC_RLR = quantile(AUC_RLR_list, 0.75)
                    quantile_75_AUC_DRO_DP = quantile(AUC_DRO_DP_list, 0.75)
                    push!(vec_dict["LR"], quantile_75_AUC_RLR)
                    push!(vec_dict["RLR"], quantile_75_AUC_RLR)
                    push!(vec_dict["DP"], quantile_75_AUC_DRO_DP)

                    
                    best_AUC_LR = maximum(AUC_LR_list)
                    best_AUC_RLR = maximum(AUC_RLR_list)
                    best_AUC_DRO_DP = maximum(AUC_DRO_DP_list)
                    push!(vec_dict["LR"], best_AUC_LR)
                    push!(vec_dict["RLR"], best_AUC_RLR)
                    push!(vec_dict["DP"], best_AUC_DRO_DP)



                    avg_AUC_LR = mean(AUC_LR_list)
                    avg_AUC_RLR = mean(AUC_RLR_list)
                    avg_AUC_DRO_DP = mean(AUC_DRO_DP_list)
                    push!(vec_dict["LR"], avg_AUC_LR)
                    push!(vec_dict["RLR"], avg_AUC_RLR)
                    push!(vec_dict["DP"], avg_AUC_DRO_DP)


                    std_AUC_LR = std(AUC_LR_list)
                    std_AUC_RLR = std(AUC_RLR_list)
                    std_AUC_DRO_DP = std(AUC_DRO_DP_list)
                    push!(vec_dict["LR"], std_AUC_LR)
                    push!(vec_dict["RLR"], std_AUC_RLR)
                    push!(vec_dict["DP"], std_AUC_DRO_DP)


                    # misclassification
                    worst_error_LR = maximum(error_LR_list)
                    worst_error_DRO_DP = maximum(error_DRO_DP_list)
                    worst_error_RLR = maximum(error_RLR_list)
                    push!(vec_dict["LR"], worst_error_LR)
                    push!(vec_dict["DP"], worst_error_DRO_DP)
                    push!(vec_dict["RLR"], worst_error_RLR)


                    quantile_25_error_LR = quantile(error_LR_list, 0.25)
                    quantile_25_error_DRO_DP = quantile(error_DRO_DP_list, 0.25)
                    quantile_25_error_RLR = quantile(error_RLR_list, 0.25)
                    push!(vec_dict["LR"], quantile_25_error_LR)
                    push!(vec_dict["DP"], quantile_25_error_DRO_DP)
                    push!(vec_dict["RLR"], quantile_25_error_RLR)

                    
                    quantile_50_error_LR = quantile(error_LR_list, 0.50)
                    quantile_50_error_DRO_DP = quantile(error_DRO_DP_list, 0.50)
                    quantile_50_error_RLR = quantile(error_RLR_list, 0.50)
                    push!(vec_dict["LR"], quantile_50_error_LR)
                    push!(vec_dict["DP"], quantile_50_error_DRO_DP)
                    push!(vec_dict["RLR"], quantile_50_error_RLR)


                    quantile_75_error_LR = quantile(error_LR_list, 0.75)
                    quantile_75_error_DRO_DP = quantile(error_DRO_DP_list, 0.75)
                    quantile_75_error_RLR = quantile(error_RLR_list, 0.75)
                    push!(vec_dict["LR"], quantile_75_error_LR)
                    push!(vec_dict["DP"], quantile_75_error_DRO_DP)
                    push!(vec_dict["RLR"], quantile_75_error_RLR)


                    best_error_LR = minimum(error_LR_list)
                    best_error_DRO_DP = minimum(error_DRO_DP_list)
                    best_error_RLR = minimum(error_RLR_list)
                    push!(vec_dict["LR"], best_error_LR)
                    push!(vec_dict["DP"], best_error_DRO_DP)
                    push!(vec_dict["RLR"], best_error_RLR)


                    avg_error_LR = mean(error_LR_list)
                    avg_error_DRO_DP = mean(error_DRO_DP_list)
                    avg_error_RLR = mean(error_RLR_list)
                    push!(vec_dict["LR"], avg_error_LR)
                    push!(vec_dict["DP"], avg_error_DRO_DP)
                    push!(vec_dict["RLR"], avg_error_RLR)


                    std_error_LR = std(error_LR_list)
                    std_error_DRO_DP = std(error_DRO_DP_list)
                    std_error_RLR = std(error_RLR_list)
                    push!(vec_dict["LR"], std_error_LR)
                    push!(vec_dict["DP"], std_error_DRO_DP)
                    push!(vec_dict["RLR"], std_error_RLR)

                end
                for method in keys(vec_dict)
                    if !(length(vec_dict[method]) == output_length)
                        print(length(vec_dict[method]))
                        print(method)
                        error("output format")
                    end
                end


                for method in ["LR", "RLR", "DP"]
                    push!(df, (dataset=vec_dict[method][1], data_type = vec_dict[method][2], method=vec_dict[method][3], lambda=vec_dict[method][4], prob_mean = vec_dict[method][5], 
                    prob_std=vec_dict[method][6], gamma_digit=vec_dict[method][7], prob_seed = vec_dict[method][8], 
                    regular_coeff = vec_dict[method][9], regular_type = vec_dict[method][10], obj=vec_dict[method][11],
                    total_time=vec_dict[method][12], total_alloc=vec_dict[method][13], solver_time=vec_dict[method][14], build_time=vec_dict[method][15], 
                    num_cut=vec_dict[method][16], avg_cut_time=vec_dict[method][17], avg_identify_table_len = vec_dict[method][18],
                    np_cal=vec_dict[method][19], np_AUC=vec_dict[method][20], np_misclassification=vec_dict[method][21], 
                    worst_cal=vec_dict[method][22], q25_cal=vec_dict[method][23], q50_cal=vec_dict[method][24], 
                    q75_cal=vec_dict[method][25], best_cal=vec_dict[method][26], avg_cal =vec_dict[method][27], std_cal = vec_dict[method][28],  
                    worst_AUC=vec_dict[method][29], q25_AUC=vec_dict[method][30], q50_AUC=vec_dict[method][31], 
                    q75_AUC=vec_dict[method][32], best_AUC=vec_dict[method][33], avg_AUC =vec_dict[method][34], std_AUC = vec_dict[method][35],  
                    worst_misclassification=vec_dict[method][36], q25_misclassification=vec_dict[method][37], q50_misclassification=vec_dict[method][38], 
                    q75_misclassification=vec_dict[method][39], best_misclassification=vec_dict[method][40], avg_misclassification =vec_dict[method][41], std_misclassification = vec_dict[method][42], 
                    worst_cal_n02 = vec_dict[method][43], q25_cal_n02 =vec_dict[method][44], q50_cal_n02 =vec_dict[method][45], 
                    q75_cal_n02 =vec_dict[method][46], best_cal_n02 =vec_dict[method][47], avg_cal_n02 =vec_dict[method][48], std_cal_n02 = vec_dict[method][49],  
                    worst_AUC_n02 =vec_dict[method][50], q25_AUC_n02 =vec_dict[method][51], q50_AUC_n02 =vec_dict[method][52], 
                    q75_AUC_n02 =vec_dict[method][53], best_AUC_n02 =vec_dict[method][54], avg_AUC_n02 =vec_dict[method][55], std_AUC_n02 = vec_dict[method][56],  
                    worst_misclassification_n02 =vec_dict[method][57], q25_misclassification_n02 =vec_dict[method][58], q50_misclassification_n02 =vec_dict[method][59], 
                    q75_misclassification_n02 =vec_dict[method][60], best_misclassification_n02 =vec_dict[method][61], avg_misclassification_n02 =vec_dict[method][62], std_misclassification_n02 = vec_dict[method][63],
                    worst_cal_n01 = vec_dict[method][64], q25_cal_n01 =vec_dict[method][65], q50_cal_n01 =vec_dict[method][66], 
                    q75_cal_n01 =vec_dict[method][67], best_cal_n01 =vec_dict[method][68], avg_cal_n01 =vec_dict[method][69], std_cal_n01 = vec_dict[method][70],  
                    worst_AUC_n01 =vec_dict[method][71], q25_AUC_n01 =vec_dict[method][72], q50_AUC_n01 =vec_dict[method][73], 
                    q75_AUC_n01 =vec_dict[method][74], best_AUC_n01 =vec_dict[method][75], avg_AUC_n01 =vec_dict[method][76], std_AUC_n01 = vec_dict[method][77],  
                    worst_misclassification_n01 =vec_dict[method][78], q25_misclassification_n01 =vec_dict[method][79], q50_misclassification_n01 =vec_dict[method][80], 
                    q75_misclassification_n01 =vec_dict[method][81], best_misclassification_n01 =vec_dict[method][82], avg_misclassification_n01 =vec_dict[method][83], std_misclassification_n01 = vec_dict[method][84],
                    worst_cal_p01 = vec_dict[method][85], q25_cal_p01 =vec_dict[method][86], q50_cal_p01 =vec_dict[method][87], 
                    q75_cal_p01 =vec_dict[method][88], best_cal_p01 =vec_dict[method][89], avg_cal_p01 =vec_dict[method][90], std_cal_p01 = vec_dict[method][91],  
                    worst_AUC_p01 =vec_dict[method][92], q25_AUC_p01 =vec_dict[method][93], q50_AUC_p01 =vec_dict[method][94], 
                    q75_AUC_p01 =vec_dict[method][95], best_AUC_p01 =vec_dict[method][96], avg_AUC_p01 =vec_dict[method][97], std_AUC_p01 = vec_dict[method][98],  
                    worst_misclassification_p01 =vec_dict[method][99], q25_misclassification_p01 =vec_dict[method][100], q50_misclassification_p01 =vec_dict[method][101], 
                    q75_misclassification_p01 =vec_dict[method][102], best_misclassification_p01 =vec_dict[method][103], avg_misclassification_p01 =vec_dict[method][104], std_misclassification_p01 = vec_dict[method][105],
                    worst_cal_r005d = vec_dict[method][106], q25_cal_r005d =vec_dict[method][107], q50_cal_r005d =vec_dict[method][108], 
                    q75_cal_r005d =vec_dict[method][109], best_cal_r005d =vec_dict[method][110], avg_cal_r005d =vec_dict[method][111], std_cal_r005d = vec_dict[method][112],  
                    worst_AUC_r005d =vec_dict[method][113], q25_AUC_r005d =vec_dict[method][114], q50_AUC_r005d =vec_dict[method][115], 
                    q75_AUC_r005d =vec_dict[method][116], best_AUC_r005d =vec_dict[method][117], avg_AUC_r005d =vec_dict[method][118], std_AUC_r005d = vec_dict[method][119],  
                    worst_misclassification_r005d =vec_dict[method][120], q25_misclassification_r005d =vec_dict[method][121], q50_misclassification_r005d =vec_dict[method][122], 
                    q75_misclassification_r005d =vec_dict[method][123], best_misclassification_r005d =vec_dict[method][124], avg_misclassification_r005d =vec_dict[method][125], std_misclassification_r005d = vec_dict[method][126],
                    worst_cal_r01d = vec_dict[method][127], q25_cal_r01d =vec_dict[method][128], q50_cal_r01d =vec_dict[method][129], 
                    q75_cal_r01d =vec_dict[method][130], best_cal_r01d =vec_dict[method][131], avg_cal_r01d =vec_dict[method][132], std_cal_r01d = vec_dict[method][133],  
                    worst_AUC_r01d =vec_dict[method][134], q25_AUC_r01d =vec_dict[method][135], q50_AUC_r01d =vec_dict[method][136], 
                    q75_AUC_r01d =vec_dict[method][137], best_AUC_r01d =vec_dict[method][138], avg_AUC_r01d =vec_dict[method][139], std_AUC_r01d = vec_dict[method][140],  
                    worst_misclassification_r01d =vec_dict[method][141], q25_misclassification_r01d =vec_dict[method][142], q50_misclassification_r01d =vec_dict[method][143], 
                    q75_misclassification_r01d =vec_dict[method][144], best_misclassification_r01d =vec_dict[method][145], avg_misclassification_r01d =vec_dict[method][146], std_misclassification_r01d = vec_dict[method][147],
                    worst_cal_r015d = vec_dict[method][148], q25_cal_r015d =vec_dict[method][149], q50_cal_r015d =vec_dict[method][150], 
                    q75_cal_r015d =vec_dict[method][151], best_cal_r015d =vec_dict[method][152], avg_cal_r015d =vec_dict[method][153], std_cal_r015d = vec_dict[method][154],  
                    worst_AUC_r015d =vec_dict[method][155], q25_AUC_r015d =vec_dict[method][156], q50_AUC_r015d =vec_dict[method][157], 
                    q75_AUC_r015d =vec_dict[method][158], best_AUC_r015d =vec_dict[method][159], avg_AUC_r015d =vec_dict[method][160], std_AUC_r015d = vec_dict[method][161],  
                    worst_misclassification_r015d =vec_dict[method][162], q25_misclassification_r015d =vec_dict[method][163], q50_misclassification_r015d =vec_dict[method][164], 
                    q75_misclassification_r015d =vec_dict[method][165], best_misclassification_r015d =vec_dict[method][166], avg_misclassification_r015d =vec_dict[method][167], std_misclassification_r015d = vec_dict[method][168],
                    worst_cal_r02d = vec_dict[method][169], q25_cal_r02d =vec_dict[method][170], q50_cal_r02d =vec_dict[method][171], 
                    q75_cal_r02d =vec_dict[method][172], best_cal_r02d =vec_dict[method][173], avg_cal_r02d =vec_dict[method][174], std_cal_r02d = vec_dict[method][175],  
                    worst_AUC_r02d =vec_dict[method][176], q25_AUC_r02d =vec_dict[method][177], q50_AUC_r02d =vec_dict[method][178], 
                    q75_AUC_r02d =vec_dict[method][179], best_AUC_r02d =vec_dict[method][180], avg_AUC_r02d =vec_dict[method][181], std_AUC_r02d = vec_dict[method][182],  
                    worst_misclassification_r02d =vec_dict[method][183], q25_misclassification_r02d =vec_dict[method][184], q50_misclassification_r02d =vec_dict[method][185], 
                    q75_misclassification_r02d =vec_dict[method][186], best_misclassification_r02d =vec_dict[method][187], avg_misclassification_r02d =vec_dict[method][188], std_misclassification_r02d = vec_dict[method][189]))
                end
                CSV.write(local_path * dataset_list[1] * "_" * data_type * "_output_dp.csv", df)
            end  
        end

    end

end
CSV.write(local_path * dataset_list[1] * "_" * data_type * "_output_dp.csv", df)
CSV.write(local_path * dataset_list[1] * "_" * data_type * "_output_dp_exceptions.csv", exception_results)





