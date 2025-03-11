# given weight parameters, generate the possible distances when we consider first k categorical features for each k \in [N] 
function generate_node_indice_list_cate(gammas)
    result_dict = Dict{Int, Vector{Float64}}()
    result_len_dict = Dict()
    
    # Start with the base case
    result_dict[0] = Float64[0]
    result_len_dict[0] = 1

    current_set = Float64[0, gammas[1]]

    result_dict[1] = current_set
    list_length = 3
    result_len_dict[1] = list_length
    for i = 2:length(gammas)
        next_set = unique(vcat(current_set, round.(current_set .+ gammas[i], digits=13)))
        sort!(next_set)
        current_set = next_set
        list_length += length(current_set)
        result_dict[i] = current_set
        result_len_dict[i] = list_length
    end
    result_dict[length(gammas)+1] = Float64[0]
    list_length += 1
    result_len_dict[length(gammas)+1] = list_length

    return result_dict, result_len_dict, list_length
end


# train DRO model
function build_DP_model_cate_oneset(X_train, y_train, groups, gammas, epsilon, dual, data_type_list, laplace_b_unbounded_vector = nothing, X_cont_train=nothing)
   # create model, add constraints and variables
    N, m = size(X_train) 
    T = length(groups)
    if dual == 0 #if solve the model by dual cone
        model = Model() 
        set_optimizer(model, MosekTools.Optimizer) 
        set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1) 
    else
        model = Model(dual_optimizer(MosekTools.Optimizer)) 
        set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1) 
    end
    @variable(model, beta[1:m]) 
    @variable(model, beta0) 
    @variable(model, 0.0 <= s[1:N]) 
    @variable(model, 0.0 <= lambda) 

    if data_type_list[1] == "cont"
        n = size(X_cont_train)[2]
        @variable(model, beta_cont[1:n]) 
        gammas_cont = 1.0 ./ laplace_b_unbounded_vector
        gammas_cont_reciprocal = 1.0 ./ gammas_cont
    
        @constraint(model,  gammas_cont_reciprocal .* beta_cont[1:n] .<= lambda )
        @constraint(model, -lambda  .<=  gammas_cont_reciprocal .* beta_cont[1:n] ) 
    end

    for g in groups
        len_g = length(g)
        if len_g >= 3
            @constraint(model, beta[g[end]] == 0) #set the coefficient of last encoded feature for each original categorical feature to 0
        end
    end

    # create variables for dual variables in the dual of longest path problems
    node_indice_list, node_indice_len_dict, node_indice_list_length = generate_node_indice_list_cate(gammas)

    mu_length = node_indice_list_length * N  
    @variable(model, mu[1:mu_length])
    @variable(model, p[1:mu_length])
    @objective(model, Min, (lambda*epsilon) + (sum(s)/N))
    
    for i in 1:N
        add = (i-1)*node_indice_list_length
        x_i = X_train[i,:]    
        y_i = y_train[i]

        if data_type_list[1] == "cont"
            x_cont_i = X_cont_train[i,:]
            @constraint(model, -mu[1+add] + mu[node_indice_list_length + add] <= y_i * beta0 + y_i * beta_cont'* x_cont_i)
        else
            @constraint(model, -mu[1+add] + mu[node_indice_list_length + add] <= y_i * beta0)
        end

        for (feature_index, group) in enumerate(groups)
            len_group = length(group)
            z_group = fill(-1, len_group)

            for (h_index, node_index) in enumerate(node_indice_len_dict[feature_index-1]+1:node_indice_len_dict[feature_index])
                arrival_index = node_index + add

                if len_group == 1
                    if node_indice_list[feature_index][h_index] in node_indice_list[feature_index-1]
                        depart_index = node_indice_len_dict[feature_index-1] - length(node_indice_list[feature_index-1])+ searchsortedfirst(node_indice_list[feature_index-1], node_indice_list[feature_index][h_index]) + add
                        @constraint(model, -mu[arrival_index] + mu[depart_index] <= y_i * beta[group]'* x_i[group])
                    end

                    if round(node_indice_list[feature_index][h_index] - gammas[feature_index],  digits=13) in node_indice_list[feature_index-1]
                        depart_index2 = node_indice_len_dict[feature_index-1] - length(node_indice_list[feature_index-1]) + searchsortedfirst(node_indice_list[feature_index-1], round(node_indice_list[feature_index][h_index] - gammas[feature_index],  digits=13)) + add
                        @constraint(model, -mu[arrival_index] + mu[depart_index2] <= y_i * beta[group]'* (-x_i[group]))
                    end 

                else
                    for j in 1:len_group
                        z_group[j] = 1
                        if z_group == x_i[group]
                            if node_indice_list[feature_index][h_index] in node_indice_list[feature_index-1]
                                depart_index = node_indice_len_dict[feature_index-1] - length(node_indice_list[feature_index-1]) + searchsortedfirst(node_indice_list[feature_index-1], node_indice_list[feature_index][h_index]) + add                                      
                                @constraint(model, -mu[arrival_index] + mu[depart_index] <= y_i * z_group'*beta[group])
                            end
                        else
                            if round(node_indice_list[feature_index][h_index] - gammas[feature_index],  digits=13) in node_indice_list[feature_index-1]
                                depart_index = node_indice_len_dict[feature_index-1] - length(node_indice_list[feature_index-1]) + searchsortedfirst(node_indice_list[feature_index-1], round(node_indice_list[feature_index][h_index] - gammas[feature_index],  digits=13)) + add
                                @constraint(model, -mu[arrival_index] + mu[depart_index] <= y_i * z_group'*beta[group])
                            end
                        end
                        z_group[j] = -1
                    end
                end
            end
        end
        sink_index = node_indice_list_length + add
        for (h_index, h) in enumerate(node_indice_list[T])
            depart_index = node_indice_len_dict[T-1] + h_index + add
            softplus(model, (s[i]+lambda*h), -(mu[sink_index] - mu[depart_index]))
        end
    end
    return model
end


function DP_wasserstein_cate_oneset(X_train, y_train, groups, gammas, epsilon, dual, data_type_list, laplace_b_unbounded_vector = nothing, X_cont_train=nothing)
#inputs:
#X_train, y_train: categorical covariates and labels in the training dataset
# groups: for each categorical feature, its range of encoded features after one hot encoding
# gammas: the weight parameters of categorical features
# epsilon: ambiguity set radius
#dual: whether we solve models by dual cone 
#data_type_list: the types of data in the dataset
# laplace_b_unbounded_vector: the scale parameter of the lapalce distribution for each numerical feature
# X_cont_train: the numerical feature part of the training dataset

    model, build_time, build_byte, build_gctime, build_m = @timed build_DP_model_cate_oneset(X_train, y_train, groups, gammas, epsilon, dual, data_type_list, laplace_b_unbounded_vector, X_cont_train)
    set_silent(model)
    JuMP.optimize!(model)

    solver_time = solve_time(model);
    return model, solver_time, build_time, build_byte
end


