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
    column_groups,
    row_groups,
    sparsity_pattern,
    decompress!
import SparseMatrixColorings as SMC

include("jacobian.jl")
include("hessian.jl")

end
