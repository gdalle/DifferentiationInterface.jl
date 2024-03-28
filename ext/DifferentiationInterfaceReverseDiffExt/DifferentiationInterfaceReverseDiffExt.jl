module DifferentiationInterfaceReverseDiffExt

using ADTypes: AutoReverseDiff
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

include("allocating.jl")
include("mutating.jl")

end # module
