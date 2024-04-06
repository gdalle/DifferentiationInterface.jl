module DifferentiationInterfaceFastDifferentiationExt

using ADTypes: ADTypes
import DifferentiationInterface as DI
using DifferentiationInterface: AutoFastDifferentiation, AutoSparseFastDifferentiation
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    HVPExtras,
    JacobianExtras,
    PullbackExtras,
    PushforwardExtras,
    SecondDerivativeExtras
using FastDifferentiation:
    derivative,
    hessian,
    hessian_times_v,
    jacobian,
    jacobian_times_v,
    jacobian_transpose_v,
    make_function,
    make_variables,
    sparse_hessian,
    sparse_jacobian
using LinearAlgebra: dot
using FastDifferentiation.RuntimeGeneratedFunctions: RuntimeGeneratedFunction

const AnyAutoFastDifferentiation = Union{
    AutoFastDifferentiation,AutoSparseFastDifferentiation
}

DI.mode(::AnyAutoFastDifferentiation) = ADTypes.AbstractSymbolicDifferentiationMode
DI.pushforward_performance(::AnyAutoFastDifferentiation) = DI.PushforwardFast()
DI.pullback_performance(::AnyAutoFastDifferentiation) = DI.PullbackSlow()

myvec(x::Number) = [x]
myvec(x::AbstractArray) = vec(x)

issparse(::AutoFastDifferentiation) = false
issparse(::AutoSparseFastDifferentiation) = true

include("allocating.jl")
include("mutating.jl")

end
