const JLVEC = jl(IVEC)
const JLMAT = jl(IMAT)

"""
    gpu_scenarios()

Create a vector of [`AbstractScenario`](@ref)s with GPU array types from [JLArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl/tree/master/lib/JLArrays).
"""
function gpu_scenarios()
    return vcat(
        # allocating
        num_to_arr_scenarios_allocating(randn(), JLVEC),
        num_to_arr_scenarios_allocating(randn(), JLMAT),
        arr_to_num_scenarios_allocating(jl(randn(6))),
        arr_to_num_scenarios_allocating(jl(randn(2, 3))),
        vec_to_vec_scenarios_allocating(jl(randn(6))),
        vec_to_mat_scenarios_allocating(jl(randn(6))),
        mat_to_vec_scenarios_allocating(jl(randn(2, 3))),
        mat_to_mat_scenarios_allocating(jl(randn(2, 3))),
        # mutating
        num_to_arr_scenarios_mutating(randn(), JLVEC),
        num_to_arr_scenarios_mutating(randn(), JLMAT),
        vec_to_vec_scenarios_mutating(jl(randn(6))),
        vec_to_mat_scenarios_mutating(jl(randn(6))),
        mat_to_vec_scenarios_mutating(jl(randn(2, 3))),
        mat_to_mat_scenarios_mutating(jl(randn(2, 3))),
    )
end
