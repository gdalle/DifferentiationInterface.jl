module DifferentiationInterfaceSparseMatrixColoringsExt

using ADTypes:
    ADTypes,
    AbstractADType,
    AutoSparse,
    dense_ad,
    coloring_algorithm,
    sparsity_detector,
    jacobian_sparsity,
    hessian_sparsity
using Compat
using DifferentiationInterface
using DifferentiationInterface:
    GradientExtras,
    HessianExtras,
    HVPExtras,
    JacobianExtras,
    PullbackExtras,
    PushforwardExtras,
    PushforwardFast,
    PushforwardSlow,
    dense_ad,
    maybe_dense_ad,
    maybe_inner,
    maybe_outer,
    multibasis,
    pick_batchsize,
    pushforward_performance,
    unwrap
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
    decompress,
    decompress!

with_context(f, contexts::Vararg{Context,C}) where {C} = (DI.with_context(f, contexts...),)

function with_context(f!, y, contexts::Vararg{Context,C}) where {C}
    return (DI.with_context(f!, contexts...), y)
end

include("jacobian.jl")
include("hessian.jl")

end
