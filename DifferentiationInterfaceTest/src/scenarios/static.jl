const SVEC = SVector{length(IVEC)}(IVEC)
const SMAT = SMatrix{size(IMAT, 1),size(IMAT, 2)}(IMAT)

"""
    static_scenarios()

Create a vector of [`AbstractScenario`](@ref)s with static array types from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
"""
function static_scenarios(; linalg=true)
    scens = vcat(
        # one argument
        num_to_arr_scenarios_onearg(rand(), SVEC),
        num_to_arr_scenarios_onearg(rand(), SMAT),
        arr_to_num_scenarios_onearg(SVector{6}(rand(6)); linalg),
        arr_to_num_scenarios_onearg(SMatrix{2,3}(rand(2, 3)); linalg),
        vec_to_vec_scenarios_onearg(SVector{6}(rand(6))),
        vec_to_mat_scenarios_onearg(SVector{6}(rand(6))),
        mat_to_vec_scenarios_onearg(SMatrix{2,3}(rand(2, 3))),
        mat_to_mat_scenarios_onearg(SMatrix{2,3}(rand(2, 3))),
        # two arguments
        num_to_arr_scenarios_twoarg(rand(), SVEC),
        num_to_arr_scenarios_twoarg(rand(), SMAT),
        vec_to_vec_scenarios_twoarg(MVector{6}(rand(6))),
        vec_to_mat_scenarios_twoarg(MVector{6}(rand(6))),
        mat_to_vec_scenarios_twoarg(MMatrix{2,3}(rand(2, 3))),
        mat_to_mat_scenarios_twoarg(MMatrix{2,3}(rand(2, 3))),
    )
    scens = filter(scens) do s
        operator_place(s) == :outofplace || s.x isa Union{Number,MVector,MMatrix}
    end
    return scens
end
