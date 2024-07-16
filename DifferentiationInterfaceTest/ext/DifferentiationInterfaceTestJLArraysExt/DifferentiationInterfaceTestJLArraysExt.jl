module DifferentiationInterfaceTestJLArraysExt

using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using JLArrays: JLArray, jl
using Random: AbstractRNG, default_rng

num_to_arr_jlvector(x) = num_to_arr(x, JLArray{Float64,1})
num_to_arr_jlmatrix(x) = num_to_arr(x, JLArray{Float64,2})

DIT.pick_num_to_arr(::Type{<:JLArray{<:Real,1}}) = num_to_arr_jlvector
DIT.pick_num_to_arr(::Type{<:JLArray{<:Real,2}}) = num_to_arr_jlmatrix

function DIT.gpu_scenarios(rng::AbstractRNG=default_rng(); linalg=true)
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

    scens = vcat(
        # one argument
        DIT.num_to_num_scenarios_onearg(x_; dx=dx_, dy=dy_),
        DIT.num_to_arr_scenarios_onearg(x_, V; dx=dx_, dy=dy_6),
        DIT.num_to_arr_scenarios_onearg(x_, M; dx=dx_, dy=dy_2_3),
        DIT.arr_to_num_scenarios_onearg(x_6; dx=dx_6, dy=dy_, linalg),
        DIT.arr_to_num_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_, linalg),
        DIT.vec_to_vec_scenarios_onearg(x_6; dx=dx_6, dy=dy_12),
        DIT.vec_to_mat_scenarios_onearg(x_6; dx=dx_6, dy=dy_6_2),
        DIT.mat_to_vec_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_12),
        DIT.mat_to_mat_scenarios_onearg(x_2_3; dx=dx_2_3, dy=dy_6_2),
        # two arguments
        DIT.num_to_arr_scenarios_twoarg(x_, V; dx=dx_, dy=dy_6),
        DIT.num_to_arr_scenarios_twoarg(x_, M; dx=dx_, dy=dy_2_3),
        DIT.vec_to_vec_scenarios_twoarg(x_6; dx=dx_6, dy=dy_12),
        DIT.vec_to_mat_scenarios_twoarg(x_6; dx=dx_6, dy=dy_6_2),
        DIT.mat_to_vec_scenarios_twoarg(x_2_3; dx=dx_2_3, dy=dy_12),
        DIT.mat_to_mat_scenarios_twoarg(x_2_3; dx=dx_2_3, dy=dy_6_2),
    )
    return scens
end

end
