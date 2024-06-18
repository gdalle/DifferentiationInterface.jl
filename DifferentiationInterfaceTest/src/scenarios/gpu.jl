"""
    gpu_scenarios(rng=Random.default_rng())

Create a vector of [`Scenario`](@ref)s with GPU array types from [JLArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl/tree/master/lib/JLArrays).
"""
function gpu_scenarios(rng::AbstractRNG=default_rng(); linalg=true)
    x_ = rand(rng)
    dx_ = rand(rng)
    dy_ = rand(rng)
    
    x_6 = jl(rand(rng, 6))
    dx_6 = jl(rand(rng, 6))
    
    x_2_3 = jl(rand(rng, 2, 3))
    dx_2_3 = jl(rand(rng, 2, 3))
    
    dy_12 = jl(rand(rng, 12))
    dy_6_2 = jl(rand(rng, 6, 2))
    dy_6 = jl(rand(rng, 6))
    dy_2_3 = jl(rand(rng, 2, 3))

    V = typeof(dy_6)
    M = typeof(dy_2_3)

    return vcat(
        # one argument
        num_to_num_scenarios_onearg(x_; dx=dx_, dy=dy_),
        num_to_arr_scenarios_onearg(x_, V; dx=dx_, dy=dy_6),
        num_to_arr_scenarios_onearg(x_, M; dx=dx_, dy=dy_2_3),
        arr_to_num_scenarios_onearg(x_6; dx=dx_6, dy=dy_, linalg),
        arr_to_num_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_, linalg),
        vec_to_vec_scenarios_onearg(x_6; dx=dx_6, dy=dy_12),
        vec_to_mat_scenarios_onearg(x_6; dx=dx_6, dy=dy_6_2),
        mat_to_vec_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_12),
        mat_to_mat_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_6_2),
        # two arguments
        num_to_arr_scenarios_twoarg(x_, V; dx=dx_, dy=dy_6),
        num_to_arr_scenarios_twoarg(x_, M; dx=dx_, dy=dy_2_3),
        vec_to_vec_scenarios_twoarg(x_6; dx=dx_6, dy=dy_12),
        vec_to_mat_scenarios_twoarg(x_6; dx=dx_6, dy=dy_6_2),
        mat_to_vec_scenarios_twoarg(x_2_3; dx=dx_2_3, dy=dy_12),
        mat_to_mat_scenarios_twoarg(x_2_3; dx=dx_2_3, dy=dy_6_2),
    )
end
