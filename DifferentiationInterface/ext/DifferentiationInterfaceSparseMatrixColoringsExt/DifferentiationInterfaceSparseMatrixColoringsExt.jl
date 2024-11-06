module DifferentiationInterfaceSparseMatrixColoringsExt

using ADTypes:
    ADTypes,
    AutoSparse,
    coloring_algorithm,
    dense_ad,
    sparsity_detector,
    jacobian_sparsity,
    hessian_sparsity
using DifferentiationInterface
using DifferentiationInterface:
    BatchSizeSettings,
    GradientPrep,
    HessianPrep,
    HVPPrep,
    JacobianPrep,
    PullbackPrep,
    PushforwardPrep,
    PushforwardFast,
    PushforwardPerformance,
    inner,
    outer,
    forward_backend,
    reverse_backend,
    multibasis,
    pick_batchsize,
    pushforward_performance,
    unwrap,
    with_contexts
import DifferentiationInterface as DI
using SparseMatrixColorings:
    AbstractColoringResult,
    ColoringProblem,
    coloring,
    column_colors,
    row_colors,
    ncolors,
    column_groups,
    row_groups,
    sparsity_pattern,
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
SMC.ncolors(prep::SparseJacobianPrep) = ncolors(prep.coloring_result)

include("jacobian.jl")
include("jacobian_mixed.jl")
include("hessian.jl")

end
