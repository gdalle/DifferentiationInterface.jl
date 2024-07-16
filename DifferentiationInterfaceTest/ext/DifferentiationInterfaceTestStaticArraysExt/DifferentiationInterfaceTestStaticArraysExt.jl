module DifferentiationInterfaceTestStaticArraysExt

using DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using Random: AbstractRNG, default_rng
using SparseArrays: SparseArrays, SparseMatrixCSC, nnz, spdiagm
using StaticArrays: MArray, MMatrix, MVector, SArray, SMatrix, SVector

num_to_arr_svector(x) = DIT.num_to_arr(x, SVector{6,Float64})
num_to_arr_smatrix(x) = DIT.num_to_arr(x, SMatrix{2,3,Float64,6})

DIT.pick_num_to_arr(::Type{<:SVector}) = num_to_arr_svector
DIT.pick_num_to_arr(::Type{<:SMatrix}) = num_to_arr_smatrix

function DIT.static_scenarios(rng::AbstractRNG=default_rng(); linalg=true)
    x_ = rand(rng)
    dx_ = rand(rng)
    dy_ = rand(rng)

    x_6 = rand(rng, 6)
    dx_6 = rand(rng, 6)

    x_2_3 = rand(rng, 2, 3)
    dx_2_3 = rand(rng, 2, 3)

    dy_6 = rand(rng, 6)
    dy_12 = rand(rng, 12)
    dy_2_3 = rand(rng, 2, 3)
    dy_6_2 = rand(rng, 6, 2)

    SV_6 = SVector{6}
    MV_6 = MVector{6}
    SV_12 = SVector{12}
    MV_12 = MVector{12}

    SM_2_3 = SMatrix{2,3}
    MM_2_3 = MMatrix{2,3}
    SM_6_2 = SMatrix{6,2}
    MM_6_2 = MMatrix{6,2}

    scens = vcat(
        # one argument
        DIT.num_to_arr_scenarios_onearg(x_, SV_6; dx=dx_, dy=SV_6(dy_6)),
        DIT.num_to_arr_scenarios_onearg(x_, SM_2_3; dx=dx_, dy=SM_2_3(dy_2_3)),
        DIT.arr_to_num_scenarios_onearg(SV_6(x_6); dx=SV_6(dx_6), dy=dy_, linalg),
        DIT.arr_to_num_scenarios_onearg(SM_2_3(x_2_3); dx=SM_2_3(dx_2_3), dy=dy_, linalg),
        DIT.vec_to_vec_scenarios_onearg(SV_6(x_6); dx=SV_6(dx_6), dy=SV_12(dy_12)),
        DIT.vec_to_mat_scenarios_onearg(SV_6(x_6); dx=SV_6(dx_6), dy=SM_6_2(dy_6_2)),
        DIT.mat_to_vec_scenarios_onearg(SM_2_3(x_2_3); dx=SM_2_3(dx_2_3), dy=SV_12(dy_12)),
        DIT.mat_to_mat_scenarios_onearg(
            SM_2_3(x_2_3); dx=SM_2_3(dx_2_3), dy=SM_6_2(dy_6_2)
        ),
        # two arguments
        DIT.num_to_arr_scenarios_twoarg(x_, MV_6; dx=dx_, dy=MV_6(dy_6)),
        DIT.num_to_arr_scenarios_twoarg(x_, MM_2_3; dx=dx_, dy=MM_2_3(dy_2_3)),
        DIT.vec_to_vec_scenarios_twoarg(MV_6(x_6); dx=MV_6(dx_6), dy=MV_12(dy_12)),
        DIT.vec_to_mat_scenarios_twoarg(MV_6(x_6); dx=MV_6(dx_6), dy=MM_6_2(dy_6_2)),
        DIT.mat_to_vec_scenarios_twoarg(MM_2_3(x_2_3); dx=MM_2_3(dx_2_3), dy=MV_12(dy_12)),
        DIT.mat_to_mat_scenarios_twoarg(
            MM_2_3(x_2_3); dx=MM_2_3(dx_2_3), dy=MM_6_2(dy_6_2)
        ),
    )
    scens = filter(scens) do s
        DIT.place(s) == :outofplace || s.x isa Union{Number,MArray}
    end
    return scens
end

end
