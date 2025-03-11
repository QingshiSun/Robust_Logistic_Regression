# sample prob of certainty for numerical features
function generate_probability_certainty_cont(n, mu, sigma)
    dist = Normal(mu, sigma)
    samples = Float64[]

    while length(samples) < n
        sample = rand(dist)
        if 0 <= sample <= 1
            push!(samples, sample)
        end
    end
    return samples
end

# calculate scale parameters of laplace distributions 
function calculate_laplace_b_unbounded(probability_certainty_cont, u_bounds)
    b_vec = [-u_bounds[i]/log(1-q) for (i,q) in enumerate(probability_certainty_cont)]
    return b_vec
end

# simulate perturbations for numerical features
function generate_laplace_perturbation(nrows, b_values, fixed_seed=nothing)

    ncols = length(b_values)
    array = zeros(nrows, ncols)

    for j = 1:ncols
        b = b_values[j]
        dist = Laplace(0, b)  # μ=0 
        array[:, j] = rand(dist, nrows)
    end
    return array
end

# Calculate the certainty interval for each numerical feature
function generate_certainty_interval_bound_normal(X_cont_train, certainty_interval_prob)
    std_devs = [std(X_cont_train[:, i]) for i in 1:size(X_cont_train, 2)]
    bounds = std_devs * certainty_interval_prob
    return bounds
end



# sample prob of certainty for categorical features
function generate_probability_certainty_cate(n, groups, mu, sigma, fixed_seed = nothing)
    dist = Normal(mu, sigma)
    T = length(groups)
    samples = Array{Float64}(undef, n, T)

    for (group_id, group) in enumerate(groups)
        if length(group) == 1
            while true
                sample = rand(dist)
                if 1/2 < sample < 1
                    samples[1, group_id] = sample
                    break
                end
            end
        else
            while true
                sample = rand(dist)
                if 1/(length(group)) < sample < 1
                    samples[1, group_id] = sample
                    break
                end
            end
        end
    end

    # Duplicate the first row as the remaining rows
    for i in 2:n
        samples[i, :] = samples[1, :]
    end
    return samples
end

# calcualte weight parameters for categorical features
function generate_cate_gammas(probability_certainty_cate_matrix, groups)
    n = size(probability_certainty_cate_matrix)[1]
    T = length(groups)
    cate_gammas = Array{Float64}(undef, 1, T)
    for j in 1:T
        if length(groups[j]) == 1
            cate_gammas[1,j] = log(probability_certainty_cate_matrix[1,j] * (length(groups[j])) / (1 - probability_certainty_cate_matrix[1,j]))
        else
            cate_gammas[1,j] = log(probability_certainty_cate_matrix[1,j] * (length(groups[j]) - 1) / (1 - probability_certainty_cate_matrix[1,j]))
        end
    end
    return cate_gammas
end


#simulate perturbed categorical features
function generate_perturbed_cate_features(probability_certainty_cate_matrix, groups, X_cate_train, fixed_seed=nothing)
    n,m = size(X_cate_train)
    X_cate_train_new = Array{Float64}(undef, n, m)
    for i in 1:n
        for (group_id, group) in enumerate(groups)
            X_cate_train_new[i,group] = flip_vector_element(X_cate_train[i,group], probability_certainty_cate_matrix[i,group_id])
        end
    end
    return X_cate_train_new
end


function flip_vector_element(vector::Vector{Float64}, p::Float64)
    # Check if the vector remains the same based on the probability p
    a = rand()
    if a < p
        return vector
    else
        if length(vector) == 1
            new_vector = [-vector[1]]
            return new_vector
        else
        # Find the index of the element that is currently 1
            current_one_index = findfirst(==(1.0), vector)
            # Get all indices except for the current '1' index
            indices = setdiff(1:length(vector), current_one_index)
            # Choose one of the other indices to swap with
            new_one_index = rand(indices)
            # Swap the elements
            new_vector = copy(vector)
            new_vector[current_one_index] = -1.0
            new_vector[new_one_index] = 1.0
            return new_vector
        end
    end
end

