module DifferentiationInterfaceReverseDiffExt

using ADTypes: AutoReverseDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    JacobianExtras,
    NoPullbackExtras,
    Tangents
using FillArrays: OneElement
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

# dot is ambiguous between TrackedArrays with different eltypes
mydot(x, y) = mapreduce(sum, *, x, y)

include("onearg.jl")
include("twoarg.jl")

end # module
