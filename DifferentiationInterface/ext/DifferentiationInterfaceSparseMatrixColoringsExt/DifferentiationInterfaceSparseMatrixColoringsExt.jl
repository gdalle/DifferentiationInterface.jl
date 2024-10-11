module DifferentiationInterfaceSparseMatrixColoringsExt

using ADTypes:
    ADTypes,
    AbstractADType,
    AutoSparse,
    coloring_algorithm,
    dense_ad,
    sparsity_detector,
    jacobian_sparsity,
    hessian_sparsity
using DifferentiationInterface
using DifferentiationInterface:
    GradientPrep,
    HessianPrep,
    HVPPrep,
    JacobianPrep,
    PullbackPrep,
    PushforwardPrep,
    PushforwardFast,
    PushforwardSlow,
    forward_backend,
    inner,
    outer,
    multibasis,
    pick_batchsize,
    pick_jacobian_batchsize,
    pushforward_performance,
    reverse_backend,
    unwrap,
    with_contexts
import DifferentiationInterface as DI
using SparseMatrixColorings:
    AbstractColoringResult,
    ColoringProblem,
    GreedyColoringAlgorithm,
    coloring,
    column_colors,
    row_colors,
    column_groups,
    row_groups,
    sparsity_pattern,
    decompress,
    decompress!
import SparseMatrixColorings as SMC

function fy_with_contexts(f, contexts::Vararg{Context,C}) where {C}
    return (with_contexts(f, contexts...),)
end

function fy_with_contexts(f!, y, contexts::Vararg{Context,C}) where {C}
    return (with_contexts(f!, contexts...), y)
end

abstract type SparseJacobianPrep <: JacobianPrep end

SMC.sparsity_pattern(prep::SparseJacobianPrep) = sparsity_pattern(prep.coloring_result)
SMC.column_colors(prep::SparseJacobianPrep) = column_colors(prep.coloring_result)
SMC.column_groups(prep::SparseJacobianPrep) = column_groups(prep.coloring_result)
SMC.row_colors(prep::SparseJacobianPrep) = row_colors(prep.coloring_result)
SMC.row_groups(prep::SparseJacobianPrep) = row_groups(prep.coloring_result)

include("jacobian.jl")
include("jacobian_mixed.jl")
include("hessian.jl")

end
