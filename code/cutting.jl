# given weight parameters, generate the possible distances when we consider first k categorical features for each k \in [N] 
function generate_h_cate(cate_gammas)
    result_dict = Dict{Int, Vector{Float64}}()
    current_set = Float64[0, cate_gammas[1]] 
    result_dict[1] = current_set
    for i = 2:length(cate_gammas)
        next_set = unique(vcat(current_set, round.(current_set .+ cate_gammas[i], digits=13)))
        sort!(next_set)
        current_set = next_set
        result_dict[i] = current_set
    end

    return result_dict
end



# given a relaxed solution, find the most violated constraint
function identify_most_violated_constraints_cate(X_cate, y, groups, rounded_cate_gammas, val_beta_cate, val_beta0, val_lambda, val_s, data_type_list, val_beta_cont=nothing, X_cont=nothing)
    max_viol = -0.01 
    N, n = size(X_cate)
    t = length(groups)
    adversarial_loc = 1
    adversarial_d = 1
    adversarial_X_cate = X_cate[1,:]
    len_group_1 = length(groups[1])

    h_dict = generate_h_cate(rounded_cate_gammas)

    for i in 1:N
        X_cate_i = X_cate[i,:]
        DP_table = ones(t,length(h_dict[t])) * Inf
        z_group_1 = fill(-1, len_group_1)
        adversarial_X_cate_table = Array{Array{Int,1},2}(undef, t, length(h_dict[t]))

        #solve the subproblems when considering the first categorical feature
        if len_group_1 == 1 # if this feature has only two possible values
            if X_cate_i[1] == 1
                DP_table[1,1] = y[i] * val_beta_cate[1] 
                DP_table[1,2] = y[i] * val_beta_cate[1] * (-1)
                adversarial_X_cate_table[1,1] = [1]
                adversarial_X_cate_table[1,2] = [-1]
            else
                DP_table[1,1] = y[i] * val_beta_cate[1] * (-1)
                DP_table[1,2] = y[i] * val_beta_cate[1]
                adversarial_X_cate_table[1,1] = [-1]
                adversarial_X_cate_table[1,2] = [1]
            end
        else
            temp_obj_list_feature_1 = []
            temp_x_list_feature_1 = []
            for j in 1:len_group_1
                z_group_1[j] = 1
                if X_cate_i[groups[1]] == z_group_1
                    DP_table[1, 1] = y[i] * val_beta_cate[groups[1]]'* X_cate_i[groups[1]]
                    adversarial_X_cate_table[1,1] = copy(X_cate_i[groups[1]])
                else
                    push!(temp_obj_list_feature_1, y[i] * val_beta_cate[groups[1]]'* z_group_1) #store the possible objective value of this subproblem given a solution
                    push!(temp_x_list_feature_1, copy(z_group_1)) # store this solution
                end
                z_group_1[j] = -1
            end
            DP_table[1, 2] = minimum(temp_obj_list_feature_1)  # find the optimal objective value
            adversarial_X_cate_table[1,2] = copy(temp_x_list_feature_1[argmin(temp_obj_list_feature_1)]) # find the optimal solution
        end


        for group_id in 2:t
            for (h_id, h) in enumerate(h_dict[group_id])
                z_group = fill(-1, length(groups[group_id]))
                temp_obj_list_feature_group_id = []
                temp_x_list_feature_group_id = []
                if length(groups[group_id]) == 1
                    if h in h_dict[group_id-1] 
                        last_index = searchsortedfirst(h_dict[group_id-1], h)
                        push!(temp_obj_list_feature_group_id, DP_table[group_id-1, last_index] + y[i] * val_beta_cate[groups[group_id]]'* X_cate_i[groups[group_id]]) #store the the possible objective value of this subproblem given a solution
                        last_x = copy(adversarial_X_cate_table[group_id-1, last_index])
                        push!(temp_x_list_feature_group_id, append!(last_x, X_cate_i[groups[group_id]])) # the possible solution to the subproblem at this state
                    end
                    if round(h - rounded_cate_gammas[group_id], digits=13) in h_dict[group_id-1]
                        last_index = searchsortedfirst(h_dict[group_id-1], round(h-rounded_cate_gammas[group_id], digits=13))
                        another_x =   (-1) .* X_cate_i[groups[group_id]]
                        push!(temp_obj_list_feature_group_id, DP_table[group_id-1, last_index] + y[i] * val_beta_cate[groups[group_id]]'* another_x)
                        last_x = copy(adversarial_X_cate_table[group_id-1, last_index])
                        push!(temp_x_list_feature_group_id, append!(last_x, another_x))
                    end
                else
                    for j in 1:length(groups[group_id])
                        z_group[j] = 1
                        if X_cate_i[groups[group_id]] == z_group
                            if h in h_dict[group_id-1]
                                last_index = searchsortedfirst(h_dict[group_id-1], h)
                                push!(temp_obj_list_feature_group_id, DP_table[group_id-1, last_index] + y[i] * val_beta_cate[groups[group_id]]'* z_group)
                                last_x = copy(adversarial_X_cate_table[group_id-1, last_index])
                                push!(temp_x_list_feature_group_id, append!(last_x, z_group))
                            end
                        else
                            if round(h - rounded_cate_gammas[group_id], digits=13) in h_dict[group_id-1]
                                last_index = searchsortedfirst(h_dict[group_id-1], round(h-rounded_cate_gammas[group_id], digits=13))
                                push!(temp_obj_list_feature_group_id, DP_table[group_id-1, last_index] + y[i] * val_beta_cate[groups[group_id]]'* z_group)
                                last_x = copy(adversarial_X_cate_table[group_id-1, last_index])
                                push!(temp_x_list_feature_group_id, append!(last_x, z_group))
                            end
                        end
                        z_group[j] = -1
                    end
                end
                DP_table[group_id, h_id] = minimum(temp_obj_list_feature_group_id)
                adversarial_X_cate_table[group_id, h_id] = copy(temp_x_list_feature_group_id[argmin(temp_obj_list_feature_group_id)])
            end
        end

        # find violations based on the optimal solutions of subproblems 
        if data_type_list[1] == "cont"
            possible_violations = log.(1 .+ exp.(-y[i] * val_beta_cont'* X_cont[i,:] .- (DP_table[t,:] .+ y[i] * val_beta0))) .-  val_lambda .*  collect(h_dict[t]) .- val_s[i]
        else
            possible_violations = log.(1 .+ exp.(- (DP_table[t,:] .+ y[i] * val_beta0))) .-  val_lambda .*  collect(h_dict[t]) .- val_s[i]
        end

        violation_i = maximum(possible_violations)
        max_index = findfirst(isequal(violation_i), possible_violations)
        max_h = h_dict[t][max_index]

        if (violation_i > max_viol)
                max_viol = violation_i
                adversarial_loc = i; 
                adversarial_d = max_h
                adversarial_X_cate = copy(adversarial_X_cate_table[t,max_index])
        end
    end
    return max_viol, adversarial_loc, adversarial_d, adversarial_X_cate, length(h_dict[t])
end


function cutting_wasserstein_cate(X_cate, y,  epsilon, groups, rounded_cate_gammas,  dual_conic, max_iteration, data_type_list, laplace_b_unbounded_vector=nothing, X_cont=nothing)
    #inputs:
#X_cate, y_train: categorical covariates and labels in the training dataset
# groups: for each categorical feature, its range of encoded features after one hot encoding
# rounded_cate_gammas: the weight parameters of categorical features
# epsilon: ambiguity set radius
#dual_conic: whether we solve models by dual cone 
# max_iteration: the max number of iterations allowed in the cutting plane method
#data_type_list: the types of data in the dataset
# laplace_b_unbounded_vector: the scale parameter of the lapalce distribution for each numerical feature
# X_cont_train: the numerical feature part of the training dataset

    solver_times = zeros(0)
    N, n = size(X_cate)
    rel_gap = 10^-8
    
    if dual_conic == 0 #dualize or not decide
        model = Model() 
        set_optimizer(model, MosekTools.Optimizer)
        set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1) #
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => rel_gap)
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_TOL_REL_GAP" => rel_gap)
    else
        model = Model(dual_optimizer(MosekTools.Optimizer)) 
        set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1) 
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => rel_gap)
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_TOL_REL_GAP" => rel_gap)
    end
    #add the variables
    @variable(model, beta[1:n]) 
    @variable(model, beta0) 

    @variable(model, 0.0 <= s[1:N]) 
    @variable(model, 0.0 <= lambda) 

    if data_type_list[1] == "cont"
        n_cont = size(X_cont)[2]
        @variable(model, beta_cont[1:n_cont]) 
        gammas_cont = 1.0 ./ laplace_b_unbounded_vector
        gammas_cont_reciprocal = 1.0 ./ gammas_cont
    
        @constraint(model,  gammas_cont_reciprocal .* beta_cont[1:n_cont] .<= lambda )
        @constraint(model, -lambda  .<=  gammas_cont_reciprocal .* beta_cont[1:n_cont])
    end

    for g in groups
        if length(g) >= 3
            @constraint(model, beta[g[end]] == 0) #set the coefficient of last encoded feature for each original categorical feature to 0
        end 
    end

    @objective(model, Min, (lambda*epsilon) + (sum(s)/N))
    #optimize once first
    set_silent(model)
    JuMP.optimize!(model)
    if termination_status(model) != OPTIMAL 
        print(termination_status(model))
        error("Solution is not optimal.")
    end
    push!(solver_times, solve_time(model))

    #identify the most violated constraints;
    iteration = 0;
    violated = 1;
    identify_time_list = []
    identify_byte_list = []
    identify_DP_col_num_list = []

    while violated == 1 #while there is still some violation in the problem
        iteration = iteration + 1 
        if iteration % 400 == 0
            println("iteration: ", iteration)
        end
        if iteration >= max_iteration
            return "too long"
        end
        if data_type_list[1] == "cont"
            identify_results, identify_time, identify_byte, identify_gctime, identify_memallocs = @timed identify_most_violated_constraints_cate(X_cate, y, 
            groups, rounded_cate_gammas, JuMP.value.(model[:beta]), JuMP.value.(model[:beta0]), JuMP.value.(model[:lambda]), JuMP.value.(model[:s]), data_type_list, JuMP.value.(model[:beta_cont]), X_cont)
        else
            identify_results, identify_time, identify_byte, identify_gctime, identify_memallocs = @timed identify_most_violated_constraints_cate(X_cate, y, 
            groups, rounded_cate_gammas, JuMP.value.(model[:beta]), JuMP.value.(model[:beta0]), JuMP.value.(model[:lambda]), JuMP.value.(model[:s]), data_type_list)
        end
        max_viol, adversarial_loc, adversarial_d, adversarial_X_cate, table_col_num = identify_results
        push!(identify_time_list, identify_time)
        push!(identify_byte_list, identify_byte)
        push!(identify_DP_col_num_list, table_col_num)

        if max_viol > rel_gap
            if data_type_list[1] == "cont"
                softplus(model, s[adversarial_loc] + lambda * adversarial_d, -(X_cont[adversarial_loc,:]'* beta_cont + adversarial_X_cate'* beta + beta0) * y[adversarial_loc]) #add the most violated constraint
            else
                softplus(model, s[adversarial_loc] + lambda * adversarial_d, -(adversarial_X_cate'* beta + beta0) * y[adversarial_loc]) 
            end
            JuMP.optimize!(model) 
            push!(solver_times, solve_time(model))
        else
            violated = 0 
        end
    end
    if termination_status(model) != OPTIMAL #warn if not optimal
        return model, iteration, sum(solver_times), mean(identify_time_list),  mean(identify_byte_list), mean(identify_DP_col_num_list)
    end
    return model, iteration, sum(solver_times), mean(identify_time_list),  mean(identify_byte_list), mean(identify_DP_col_num_list)
end

function softplus_updated(model, t, linear_transform)
    z = @variable(model, [1:2], lower_bound = 0.0)
    cn1 = @constraint(model, sum(z) <= 1.0)
    cn2 = @constraint(model, [linear_transform - t, 1, z[1]] in MOI.ExponentialCone())
    cn3 = @constraint(model, [-t, 1, z[2]] in MOI.ExponentialCone())
    return z,cn1,cn2,cn3
end


function most_violated_feature_label_metric(X, y, groups, val_beta, val_beta0, val_s, val_lambda, data_type_list, val_beta_cont=nothing, X_cont=nothing)
    max_viol = -0.1 
    N, n = size(X)
    T = length(groups) 
    singletons = [k[1] for k in groups if length(k) == 1] 
    #initiate the solutions to return
    adversarial_x = copy(X[1, :]); 
    adversarial_y = 1; 
    adversarial_d = 1.0;
    adversarial_loc = 1; 
        for i = 1:N 
            x_hat = X[i, :] 
            y_hat = y[i] 
            for case in [1]
                sol1_y = y_hat*case 
                best_z= []
                for group in groups
                    if length(group) == 1 
                        best_z = vcat(best_z, -x_hat[group])
                    else
                        z_group = fill(-1, length(group))
                        best_z_group = []
                        best_product = Inf
                        for j in 1:length(z_group)
                            z_group[j] = 1 
                            if z_group == x_hat[group]
                                z_group[j] = -1
                                continue
                            end
                            product = sol1_y * val_beta[group]'*z_group
                            if product < best_product
                                best_z_group = copy(z_group)
                                best_product = product
                            end
                            z_group[j] = -1
                        end
                        best_z= vcat(best_z, best_z_group)
                    end
                end

                product_list = []
                for (group_id, group) in enumerate(groups)
                    push!(product_list, sol1_y*val_beta[group]'*(best_z[group] - x_hat[group]))    
                end
                sorted_groups_index = sortperm(product_list)
                sorted_groups = groups[sorted_groups_index]
                for diff_num in 0:length(groups)
                    if diff_num == 0
                        sol1_x = x_hat
                    else
                        sol1_x = copy(x_hat)
                        for k in 1:diff_num
                            sol1_x[sorted_groups[k]] = best_z[sorted_groups[k]]
                        end
                    end
                    sol1_d = diff_num 
                    if data_type_list[1] == "cont"
                        violation = log(1 + exp(-sol1_y * (X_cont[i,:]'* val_beta_cont + sol1_x'*val_beta + val_beta0))) - (val_lambda* sol1_d) - val_s[i];
                    else
                        violation = log(1 + exp(-sol1_y * (sol1_x'*val_beta + val_beta0))) - (val_lambda* sol1_d) - val_s[i];
                    end
                    if violation > max_viol
                        max_viol = copy(violation)
                        adversarial_x = copy(sol1_x); 
                        adversarial_y = copy(sol1_y); 
                        adversarial_d = copy(sol1_d);
                        adversarial_loc = copy(i); 
                    end
                end            
            end
        end
    return max_viol, adversarial_x, adversarial_y, adversarial_d, adversarial_loc
end


function cutting_wasserstein_selvi(X, y, groups, epsilon, dual_conic, max_iteration, data_type_list, laplace_b_unbounded_vector=nothing, X_cont=nothing)
    #inputs:
#X, y: categorical covariates and labels in the training dataset
# groups: for each categorical feature, its range of encoded features after one hot encoding
# gammas: the weight parameters of categorical features
# epsilon: ambiguity set radius
#dual: whether we solve models by dual cone 
# max_iteration: the max number of iterations allowed in the cutting plane method
#data_type_list: the types of data in the dataset
# laplace_b_unbounded_vector: the scale parameter of the lapalce distribution for each numerical feature
# X_cont: the numerical feature part of the training dataset

    
    solver_times = zeros(0)
    N, n = size(X) 
    rel_gap = 10^-8 
    if dual_conic == 0 
        model = Model() 
        set_optimizer(model, MosekTools.Optimizer)
        set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1) 
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => rel_gap)
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_TOL_REL_GAP" => rel_gap)
    else
        model = Model(dual_optimizer(MosekTools.Optimizer)) 
        set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1)
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => rel_gap)
        set_optimizer_attributes(model, "MSK_DPAR_INTPNT_TOL_REL_GAP" => rel_gap)
    end
    #add the variables and some constraints
    @variable(model, beta[1:n]) 
    @variable(model, beta0) 
    @variable(model, 0.0 <= s[1:N]) 
    @variable(model, 0.0 <= lambda) 

    if data_type_list[1] == "cont"
        n_cont = size(X_cont)[2]
        @variable(model, beta_cont[1:n_cont])
        gammas_cont = 1.0 ./ laplace_b_unbounded_vector
        gammas_cont_reciprocal = 1.0 ./ gammas_cont
    
        @constraint(model,  gammas_cont_reciprocal .* beta_cont[1:n_cont] .<= lambda )
        @constraint(model, -lambda  .<=  gammas_cont_reciprocal .* beta_cont[1:n_cont])
    end

    for g in groups
        if length(g) >= 3
            @constraint(model, beta[g[end]] == 0) #set the coefficient of last encoded feature for each original categorical feature to 0
        end 
    end

    @objective(model, Min, (lambda*epsilon) + (sum(s)/N))
    #optimize the model to get the first relaxed solution
    set_silent(model)
    JuMP.optimize!(model)
    if termination_status(model) != OPTIMAL 
        error("Solution is not optimal.")
    end
    push!(solver_times, solve_time(model))

    # following the implementation of this cutting plane method, we delete some slacked constraints periodically
    iteration = 0;
    violated = 1;
    identify_time_list = []
    vars, c1s, c2s, c3s = vec([]),vec([]),vec([]),vec([]) 
    base_to_take = 100 
    slack_to_take = 0.99 
    while violated == 1 
        iteration = iteration + 1 
        if iteration % 400 == 0
            println("iteraiton = ", iteration)
        end
        if iteration >= max_iteration #check abnormal case 
            return "too long"
        end

        if data_type_list[1] == "cont"
            identify_results, identify_time, identify_byte, identify_gctime, identify_other = @timed most_violated_feature_label_metric(X, y, groups, JuMP.value.(model[:beta]), JuMP.value.(model[:beta0]), JuMP.value.(model[:s]), JuMP.value.(model[:lambda]), data_type_list, JuMP.value.(model[:beta_cont]), X_cont)
        else
            identify_results, identify_time, identify_byte, identify_gctime, identify_other = @timed most_violated_feature_label_metric(X, y, groups, JuMP.value.(model[:beta]), JuMP.value.(model[:beta0]), JuMP.value.(model[:s]), JuMP.value.(model[:lambda]), data_type_list)
        end

        
        max_viol, adversarial_x, adversarial_y, adversarial_d, adversarial_loc = identify_results
        push!(identify_time_list, identify_time)
        if max_viol > rel_gap #if violation is still non-zero
            #add the constraint
            if data_type_list[1] == "cont"
                var, c1, c2, c3 = softplus_updated(model, s[adversarial_loc] + (lambda*adversarial_d), -(X_cont[adversarial_loc,:]'* beta_cont + (adversarial_x' * beta) + beta0) * adversarial_y) #add the most violated constraint
            else
                var, c1, c2, c3 = softplus_updated(model, s[adversarial_loc] + (lambda*adversarial_d), -((adversarial_x' * beta) + beta0) * adversarial_y) #add the most violated constraint
            end
            
            push!(vars, var)
            push!(c1s, c1)
            push!(c2s, c2)
            push!(c3s, c3)
            #optimize
            JuMP.optimize!(model)
            push!(solver_times, solve_time(model)) 
            if iteration == Int(base_to_take)
                base_to_take = round(Int64, 1.5*base_to_take) #delete constarints less often, keep updating -- prevents possible loops
                indexes_to_del = vec([]) #constarints to delete
                for (c_ind, c_to_del) in enumerate(c1s) #go over every constraint we have
                    if value(c_to_del) <= slack_to_take #if slack, delete
                        push!(indexes_to_del, c_ind)
                    end
                end
                slack_to_take = max(0.0, slack_to_take - 0.01) #keep reducing the threshold so we are less strict -- prevents possible loops
                for ind in indexes_to_del
                    delete(model, vars[ind])
                    delete(model, c1s[ind])
                    delete(model, c2s[ind])
                    delete(model, c3s[ind])
                end
                #update the vector of all constraints and remove the deleted ones
                deleteat!(vars, indexes_to_del)
                deleteat!(c1s, indexes_to_del)
                deleteat!(c2s, indexes_to_del)
                deleteat!(c3s, indexes_to_del)
                iteration = iteration +  1 
                JuMP.optimize!(model) 
                push!(solver_times, solve_time(model))
            end
        else
            violated = 0
        end
    end
    if termination_status(model) != OPTIMAL 
        return model, iteration, sum(solver_times), mean(identify_time_list)
    end
    return model, iteration, sum(solver_times), mean(identify_time_list) 
end
