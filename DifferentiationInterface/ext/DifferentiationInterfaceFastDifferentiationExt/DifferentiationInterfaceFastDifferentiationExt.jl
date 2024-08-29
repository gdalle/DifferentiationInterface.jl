module DifferentiationInterfaceFastDifferentiationExt

using ADTypes: ADTypes, AutoFastDifferentiation, AutoSparse
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    HVPExtras,
    JacobianExtras,
    PullbackExtras,
    PushforwardExtras,
    SecondDerivativeExtras,
    Tangents,
    maybe_dense_ad
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
using FillArrays: Fill
using LinearAlgebra: dot
using FastDifferentiation.RuntimeGeneratedFunctions: RuntimeGeneratedFunction

DI.check_available(::AutoFastDifferentiation) = true

monovec(x::Number) = Fill(x, 1)

myvec(x::Number) = monovec(x)
myvec(x::AbstractArray) = vec(x)

include("onearg.jl")
include("twoarg.jl")

end
