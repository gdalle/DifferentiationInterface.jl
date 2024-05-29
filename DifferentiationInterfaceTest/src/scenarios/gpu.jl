const JLVEC = jl(IVEC)
const JLMAT = jl(IMAT)

"""
    gpu_scenarios()

Create a vector of [`AbstractScenario`](@ref)s with GPU array types from [JLArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl/tree/master/lib/JLArrays).
"""
function gpu_scenarios()
    return vcat(
        # one argument
        num_to_arr_scenarios_onearg(rand(), JLVEC),
        num_to_arr_scenarios_onearg(rand(), JLMAT),
        arr_to_num_scenarios_onearg(jl(rand(6))),
        arr_to_num_scenarios_onearg(jl(rand(2, 3))),
        vec_to_vec_scenarios_onearg(jl(rand(6))),
        vec_to_mat_scenarios_onearg(jl(rand(6))),
        mat_to_vec_scenarios_onearg(jl(rand(2, 3))),
        mat_to_mat_scenarios_onearg(jl(rand(2, 3))),
        # two arguments
        num_to_arr_scenarios_twoarg(rand(), JLVEC),
        num_to_arr_scenarios_twoarg(rand(), JLMAT),
        vec_to_vec_scenarios_twoarg(jl(rand(6))),
        vec_to_mat_scenarios_twoarg(jl(rand(6))),
        mat_to_vec_scenarios_twoarg(jl(rand(2, 3))),
        mat_to_mat_scenarios_twoarg(jl(rand(2, 3))),
    )
end
