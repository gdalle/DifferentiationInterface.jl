"""
    gpu_scenarios(rng=Random.default_rng())

Create a vector of [`Scenario`](@ref)s with GPU array types from [JLArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl/tree/master/lib/JLArrays).
"""
function gpu_scenarios(rng::AbstractRNG=default_rng(); linalg=true)
    v = jl(float.(Vector(1:6)))
    m = jl(float.(Matrix((1:2) .* transpose(1:3))))

    x_ = rand(rng)
    dx_ = rand(rng)
    dy_ = rand(rng)

    x_6 = jl(rand(rng, 6))
    dx_6 = jl(rand(rng, 6))

    x_2_3 = jl(rand(rng, 2, 3))
    dx_2_3 = jl(rand(rng, 2, 3))

    dy_12 = jl(rand(rng, 12))
    dy_6_2 = jl(rand(rng, 6, 2))

    dy_v = jl(rand!(rng, similar(v)))
    dy_m = jl(rand!(rng, similar(m)))

    return vcat(
        # one argument
        num_to_num_scenarios_onearg(x_; dx=dx_, dy=dy_),
        num_to_arr_scenarios_onearg(x_, jl(v); dx=dx_, dy=dy_v),
        num_to_arr_scenarios_onearg(x_, jl(m); dx=dx_, dy=dy_m),
        arr_to_num_scenarios_onearg(x_6; dx=dx_6, dy=dy_, linalg),
        arr_to_num_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_, linalg),
        vec_to_vec_scenarios_onearg(x_6; dx=dx_6, dy=dy_12),
        vec_to_mat_scenarios_onearg(x_6; dx=dx_6, dy=dy_6_2),
        mat_to_vec_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_12),
        mat_to_mat_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_6_2),
        # two arguments
        num_to_arr_scenarios_twoarg(x_, jl(v); dx=dx_, dy=dy_v),
        num_to_arr_scenarios_twoarg(x_, jl(m); dx=dx_, dy=dy_m),
        vec_to_vec_scenarios_twoarg(x_6; dx=dx_6, dy=dy_12),
        vec_to_mat_scenarios_twoarg(x_6; dx=dx_6, dy=dy_6_2),
        mat_to_vec_scenarios_twoarg(x_2_3; dx=dx_2_3, dy=dy_12),
        mat_to_mat_scenarios_twoarg(x_2_3; dx=dx_2_3, dy=dy_6_2),
    )
end
