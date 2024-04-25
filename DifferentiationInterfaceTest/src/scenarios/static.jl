const SVEC = SVector{length(IVEC)}(IVEC)
const SMAT = SMatrix{size(IMAT, 1),size(IMAT, 2)}(IMAT)

"""
    static_scenarios()

Create a vector of [`AbstractScenario`](@ref)s with static array types from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
"""
function static_scenarios()
    scens = vcat(
        # one argument
        num_to_arr_scenarios_onearg(randn(), SVEC),
        num_to_arr_scenarios_onearg(randn(), SMAT),
        arr_to_num_scenarios_onearg(SVector{6}(randn(6))),
        arr_to_num_scenarios_onearg(SMatrix{2,3}(randn(2, 3))),
        vec_to_vec_scenarios_onearg(SVector{6}(randn(6))),
        vec_to_mat_scenarios_onearg(SVector{6}(randn(6))),
        mat_to_vec_scenarios_onearg(SMatrix{2,3}(randn(2, 3))),
        mat_to_mat_scenarios_onearg(SMatrix{2,3}(randn(2, 3))),
        # two arguments
        num_to_arr_scenarios_twoarg(randn(), SVEC),
        num_to_arr_scenarios_twoarg(randn(), SMAT),
        vec_to_vec_scenarios_twoarg(MVector{6}(randn(6))),
        vec_to_mat_scenarios_twoarg(MVector{6}(randn(6))),
        mat_to_vec_scenarios_twoarg(MMatrix{2,3}(randn(2, 3))),
        mat_to_mat_scenarios_twoarg(MMatrix{2,3}(randn(2, 3))),
    )
    scens = filter(scens) do s
        operator_place(s) == :outofplace || s.x isa Union{Number,MVector,MMatrix}
    end
    return scens
end
