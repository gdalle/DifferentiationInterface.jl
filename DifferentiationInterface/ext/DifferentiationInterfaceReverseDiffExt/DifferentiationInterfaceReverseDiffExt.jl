module DifferentiationInterfaceReverseDiffExt

using ADTypes: AutoReverseDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativePrep, GradientPrep, HessianPrep, JacobianPrep, NoPullbackPrep
using ReverseDiff.DiffResults: DiffResults, DiffResult, GradientResult, MutableDiffResult
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

DI.check_available(::AutoReverseDiff) = true

function DI.basis(::AutoReverseDiff, a::AbstractArray{T}, i) where {T}
    return DI.OneElement(i, one(T), a)
end

include("onearg.jl")
include("twoarg.jl")

end # module
