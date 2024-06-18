"""
    static_scenarios(rng=Random.default_rng())

Create a vector of [`Scenario`](@ref)s with static array types from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).
"""
function static_scenarios(rng::AbstractRNG=default_rng(); linalg=true)
    v = float.(Vector(1:6))
    m = float.(Matrix((1:2) .* transpose(1:3)))

    x_ = rand(rng)
    dx_ = rand(rng)
    dy_ = rand(rng)

    x_6 = rand(rng, 6)
    dx_6 = rand(rng, 6)

    x_2_3 = rand(rng, 2, 3)
    dx_2_3 = rand(rng, 2, 3)

    dy_12 = rand(rng, 12)
    dy_6_2 = rand(rng, 6, 2)

    dy_v = mycopy_random(rng, v)
    dy_m = mycopy_random(rng, m)

    scens = vcat(
        # one argument
        num_to_arr_scenarios_onearg(x_, SVector{6}(v); dx=dx_, dy=SVector{6}(v)),
        num_to_arr_scenarios_onearg(x_, SMatrix{2,3}(m); dx=dx_, dy=SMatrix{2,3}(dy_m)),
        arr_to_num_scenarios_onearg(SVector{6}(x_6); dx=SVector{6}(dx_6), dy=dy_, linalg),
        arr_to_num_scenarios_onearg(
            SMatrix{2,3}(x_2_3); dx=SMatrix{2,3}(dx_2_3), dy=dy_, linalg
        ),
        vec_to_vec_scenarios_onearg(
            SVector{6}(x_6); dx=SVector{6}(dx_6), dy=SVector{12}(dy_12)
        ),
        vec_to_mat_scenarios_onearg(
            SVector{6}(x_6); dx=SVector{6}(dx_6), dy=SMatrix{6,2}(dy_6_2)
        ),
        mat_to_vec_scenarios_onearg(
            SMatrix{2,3}(x_2_3); dx=SMatrix{2,3}(dx_2_3), dy=SVector{12}(dy_12)
        ),
        mat_to_mat_scenarios_onearg(
            SMatrix{2,3}(x_2_3); dx=SMatrix{2,3}(dx_2_3), dy=SMatrix{6,2}(dy_6_2)
        ),
        # two arguments
        num_to_arr_scenarios_twoarg(x_, MVector{6}(v); dx=dx_, dy=MVector{6}(dy_v)),
        num_to_arr_scenarios_twoarg(x_, MMatrix{2,3}(m); dx=dx_, dy=MMatrix{2,3}(dy_m)),
        vec_to_vec_scenarios_twoarg(
            MVector{6}(x_6); dx=MVector{6}(dx_6), dy=MVector{12}(dy_12)
        ),
        vec_to_mat_scenarios_twoarg(
            MVector{6}(x_6); dx=MVector{6}(dx_6), dy=MMatrix{6,2}(dy_6_2)
        ),
        mat_to_vec_scenarios_twoarg(
            MMatrix{2,3}(x_2_3); dx=MMatrix{2,3}(dx_2_3), dy=MVector{12}(dy_12)
        ),
        mat_to_mat_scenarios_twoarg(
            MMatrix{2,3}(x_2_3); dx=MMatrix{2,3}(dx_2_3), dy=MMatrix{6,2}(dy_6_2)
        ),
    )
    scens = filter(scens) do s
        place(s) == :outofplace || s.x isa Union{Number,MArray}
    end
    return scens
end
