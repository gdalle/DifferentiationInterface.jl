module DifferentiationInterfaceSparseMatrixColoringsExt

using ADTypes: AutoSparse, dense_ad
using ADTypes: coloring_algorithm, sparsity_detector, jacobian_sparsity, hessian_sparsity
using Compat
using DifferentiationInterface
using DifferentiationInterface:
    Batch,
    GradientExtras,
    JacobianExtras,
    HessianExtras,
    HVPExtras,
    hvp_batched,
    make_seed,
    maybe_inner,
    pick_batchsize,
    pushforward_batched,
    pullback_batched,
    prepare_gradient,
    prepare_hvp_batched_same_point,
    prepare_pushforward_batched_same_point,
    prepare_pullback_batched_same_point
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

include("fallbacks.jl")
include("jacobian.jl")
include("hessian.jl")

end
