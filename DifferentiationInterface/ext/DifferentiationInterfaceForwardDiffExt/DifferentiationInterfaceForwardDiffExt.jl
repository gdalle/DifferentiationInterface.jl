module DifferentiationInterfaceForwardDiffExt

using ADTypes: AbstractADType, AutoForwardDiff
using Base: Fix1, Fix2
import DifferentiationInterface as DI
using DifferentiationInterface:
    BatchSizeSettings,
    Context,
    DerivativePrep,
    DifferentiateWith,
    GradientPrep,
    HessianPrep,
    HVPPrep,
    JacobianPrep,
    NoDerivativePrep,
    NoSecondDerivativePrep,
    PushforwardPrep,
    Rewrap,
    SecondOrder,
    inner,
    outer,
    unwrap,
    with_contexts
import ForwardDiff.DiffResults as DR
using ForwardDiff.DiffResults:
    DiffResults, DiffResult, GradientResult, HessianResult, MutableDiffResult
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
    npartials,
    partials,
    value
using LinearAlgebra: dot, mul!

DI.check_available(::AutoForwardDiff) = true

include("utils.jl")
include("onearg.jl")
include("twoarg.jl")
include("secondorder.jl")
include("differentiate_with.jl")

end # module
