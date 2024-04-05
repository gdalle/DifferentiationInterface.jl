module DifferentiationInterfaceReverseDiffExt

using ADTypes: AutoReverseDiff, AutoSparseReverseDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras, GradientExtras, HessianExtras, JacobianExtras, NoPullbackExtras
using ReverseDiff.DiffResults: DiffResults, DiffResult, GradientResult
using DocStringExtensions
using LinearAlgebra: dot, mul!
using ReverseDiff:
    CompiledGradient,
    CompiledHessian,
    CompiledJacobian,
    GradientConfig,
    GradientTape,
    HessianConfig,
    HessianTape,
    JacobianConfig,
    JacobianTape,
    compile,
    gradient,
    gradient!,
    hessian,
    hessian!,
    jacobian,
    jacobian!

const AnyAutoReverseDiff = Union{AutoReverseDiff,AutoSparseReverseDiff}

include("allocating.jl")
include("mutating.jl")

end # module
