const JLVEC = jl(IVEC)
const JLMAT = jl(IMAT)

"""
    gpu_scenarios(rng=Random.default_rng())

Create a vector of [`AbstractScenario`](@ref)s with GPU array types from [JLArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl/tree/master/lib/JLArrays).
"""
function gpu_scenarios(rng::AbstractRNG=default_rng(); linalg=true)
    return vcat(
        # one argument
        num_to_arr_scenarios_onearg(rand(rng), JLVEC),
        num_to_arr_scenarios_onearg(rand(rng), JLMAT),
        arr_to_num_scenarios_onearg(jl(rand(rng, 6)); linalg),
        arr_to_num_scenarios_onearg(jl(rand(rng, 2, 3)); linalg),
        vec_to_vec_scenarios_onearg(jl(rand(rng, 6))),
        vec_to_mat_scenarios_onearg(jl(rand(rng, 6))),
        mat_to_vec_scenarios_onearg(jl(rand(rng, 2, 3))),
        mat_to_mat_scenarios_onearg(jl(rand(rng, 2, 3))),
        # two arguments
        num_to_arr_scenarios_twoarg(rand(rng), JLVEC),
        num_to_arr_scenarios_twoarg(rand(rng), JLMAT),
        vec_to_vec_scenarios_twoarg(jl(rand(rng, 6))),
        vec_to_mat_scenarios_twoarg(jl(rand(rng, 6))),
        mat_to_vec_scenarios_twoarg(jl(rand(rng, 2, 3))),
        mat_to_mat_scenarios_twoarg(jl(rand(rng, 2, 3))),
    )
end
