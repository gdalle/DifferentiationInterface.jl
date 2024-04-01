const SVEC = SVector{length(IVEC)}(IVEC)
const SMAT = SMatrix{size(IMAT, 1),size(IMAT, 2)}(IMAT)

"""
    static_scenarios()

Create a vector of [`AbstractScenario`](@ref)s with static array types from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
"""
function static_scenarios()
    return vcat(
        # allocating
        num_to_arr_scenarios_allocating(randn(), SVEC),
        num_to_arr_scenarios_allocating(randn(), SMAT),
        arr_to_num_scenarios_allocating(SVector{6}(randn(6))),
        arr_to_num_scenarios_allocating(SMatrix{2,3}(randn(2, 3))),
        vec_to_vec_scenarios_allocating(SVector{6}(randn(6))),
        vec_to_mat_scenarios_allocating(SVector{6}(randn(6))),
        mat_to_vec_scenarios_allocating(SMatrix{2,3}(randn(2, 3))),
        mat_to_mat_scenarios_allocating(SMatrix{2,3}(randn(2, 3))),
        # mutating
        num_to_arr_scenarios_mutating(randn(), SVEC),
        num_to_arr_scenarios_mutating(randn(), SMAT),
        vec_to_vec_scenarios_mutating(MVector{6}(randn(6))),
        vec_to_mat_scenarios_mutating(MVector{6}(randn(6))),
        mat_to_vec_scenarios_mutating(MMatrix{2,3}(randn(2, 3))),
        mat_to_mat_scenarios_mutating(MMatrix{2,3}(randn(2, 3))),
    )
end
