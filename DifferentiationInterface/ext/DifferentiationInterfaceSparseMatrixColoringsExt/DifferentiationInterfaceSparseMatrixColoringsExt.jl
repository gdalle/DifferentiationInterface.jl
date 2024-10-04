module DifferentiationInterfaceSparseMatrixColoringsExt

using ADTypes:
    ADTypes,
    AbstractADType,
    AutoSparse,
    coloring_algorithm,
    sparsity_detector,
    jacobian_sparsity,
    hessian_sparsity
using Compat
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
    dense_ad,
    inner,
    outer,
    multibasis,
    pick_batchsize,
    pick_jacobian_batchsize,
    pushforward_performance,
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

include("jacobian.jl")
include("hessian.jl")

end
