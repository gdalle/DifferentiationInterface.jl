module DifferentiationInterfaceReverseDiffExt

using ADTypes: AutoReverseDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras, GradientExtras, HessianExtras, JacobianExtras, NoPullbackExtras
using ReverseDiff.DiffResults: DiffResults, DiffResult, GradientResult, MutableDiffResult
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

DI.check_available(::AutoReverseDiff) = true

function mydot(a, b)
    s = zero(Base.promote_eltype(a, b))
    for i in eachindex(a, b)
        s += a[i] * b[i]
    end
    return s
end

include("onearg.jl")
include("twoarg.jl")

end # module
