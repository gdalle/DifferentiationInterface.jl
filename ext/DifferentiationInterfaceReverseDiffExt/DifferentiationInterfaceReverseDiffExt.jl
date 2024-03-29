module DifferentiationInterfaceReverseDiffExt

using ADTypes: AutoReverseDiff, AutoSparseReverseDiff
import DifferentiationInterface as DI
using ReverseDiff.DiffResults: DiffResults, DiffResult, GradientResult
using DocStringExtensions
using LinearAlgebra: mul!
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
