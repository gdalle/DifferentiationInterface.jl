module DifferentiationInterfaceSymbolicsExt

using ADTypes: ADTypes, AutoSymbolics, AutoSparse
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    HVPExtras,
    JacobianExtras,
    PullbackExtras,
    PushforwardExtras,
    SecondDerivativeExtras
using FillArrays: Fill
using LinearAlgebra: dot
using Symbolics:
    build_function,
    derivative,
    gradient,
    hessian,
    jacobian,
    sparsehessian,
    sparsejacobian,
    substitute,
    variable,
    variables
using Symbolics.RuntimeGeneratedFunctions: RuntimeGeneratedFunction

DI.check_available(::AutoSymbolics) = true
DI.pullback_performance(::AutoSymbolics) = DI.PullbackSlow()

monovec(x::Number) = Fill(x, 1)

myvec(x::Number) = monovec(x)
myvec(x::AbstractArray) = vec(x)

include("onearg.jl")
include("twoarg.jl")
include("detector.jl")

end
