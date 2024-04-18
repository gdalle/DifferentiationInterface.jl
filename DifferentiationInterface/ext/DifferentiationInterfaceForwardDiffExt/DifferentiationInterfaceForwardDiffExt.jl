module DifferentiationInterfaceForwardDiffExt

using ADTypes: AbstractADType, AutoForwardDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    JacobianExtras,
    NoDerivativeExtras,
    NoPushforwardExtras
using ForwardDiff.DiffResults: DiffResults, DiffResult, GradientResult
using ForwardDiff:
    Chunk,
    Dual,
    DerivativeConfig,
    ForwardDiff,
    GradientConfig,
    HessianConfig,
    JacobianConfig,
    Tag,
    derivative,
    derivative!,
    extract_derivative,
    extract_derivative!,
    gradient,
    gradient!,
    hessian,
    hessian!,
    jacobian,
    jacobian!,
    value
using LinearAlgebra: dot, mul!

DI.check_available(::AutoForwardDiff) = true

include("utils.jl")
include("onearg.jl")
include("twoarg.jl")

end # module
