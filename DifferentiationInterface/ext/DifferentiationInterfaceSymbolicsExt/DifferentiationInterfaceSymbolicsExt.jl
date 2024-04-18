module DifferentiationInterfaceSymbolicsExt

using ADTypes: ADTypes
import DifferentiationInterface as DI
using DifferentiationInterface: AutoSymbolics, AutoSparseSymbolics
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

const AnyAutoSymbolics = Union{AutoSymbolics,AutoSparseSymbolics}

DI.check_available(::AutoSymbolics) = true
DI.pullback_performance(::AutoSymbolics) = DI.PullbackSlow()

monovec(x::Number) = Fill(x, 1)

myvec(x::Number) = monovec(x)
myvec(x::AbstractArray) = vec(x)

issparse(::AutoSymbolics) = false
issparse(::AutoSparseSymbolics) = true

include("onearg.jl")
include("twoarg.jl")

end
