const SVEC = SVector{length(IVEC)}(IVEC)
const SMAT = SMatrix{size(IMAT, 1),size(IMAT, 2)}(IMAT)

"""
    static_scenarios(rng=Random.default_rng())

Create a vector of [`AbstractScenario`](@ref)s with static array types from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
"""
function static_scenarios(rng::AbstractRNG=default_rng(); linalg=true)
    scens = vcat(
        # one argument
        num_to_arr_scenarios_onearg(rand(rng), SVEC),
        num_to_arr_scenarios_onearg(rand(rng), SMAT),
        arr_to_num_scenarios_onearg(SVector{6}(rand(rng, 6)); linalg),
        arr_to_num_scenarios_onearg(SMatrix{2,3}(rand(rng, 2, 3)); linalg),
        vec_to_vec_scenarios_onearg(SVector{6}(rand(rng, 6))),
        vec_to_mat_scenarios_onearg(SVector{6}(rand(rng, 6))),
        mat_to_vec_scenarios_onearg(SMatrix{2,3}(rand(rng, 2, 3))),
        mat_to_mat_scenarios_onearg(SMatrix{2,3}(rand(rng, 2, 3))),
        # two arguments
        num_to_arr_scenarios_twoarg(rand(rng), SVEC),
        num_to_arr_scenarios_twoarg(rand(rng), SMAT),
        vec_to_vec_scenarios_twoarg(MVector{6}(rand(rng, 6))),
        vec_to_mat_scenarios_twoarg(MVector{6}(rand(rng, 6))),
        mat_to_vec_scenarios_twoarg(MMatrix{2,3}(rand(rng, 2, 3))),
        mat_to_mat_scenarios_twoarg(MMatrix{2,3}(rand(rng, 2, 3))),
    )
    scens = filter(scens) do s
        operator_place(s) == :outofplace || s.x isa Union{Number,MVector,MMatrix}
    end
    return scens
end
