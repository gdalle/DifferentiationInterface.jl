module DifferentiationInterfaceForwardDiffExt

using ADTypes: AutoForwardDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    BatchSizeSettings,
    Cache,
    Constant,
    PrepContext,
    Context,
    DerivativePrep,
    DifferentiateWith,
    GradientPrep,
    HessianPrep,
    HVPPrep,
    JacobianPrep,
    NoSecondDerivativePrep,
    PushforwardPrep,
    Rewrap,
    SecondOrder,
    inner,
    outer,
    shuffled_gradient,
    unwrap,
    with_contexts,
    ismutable_array
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
    gradient,
    gradient!,
    hessian,
    hessian!,
    jacobian,
    jacobian!,
    partials,
    value

DI.check_available(::AutoForwardDiff) = true

include("utils.jl")
include("onearg.jl")
include("twoarg.jl")
include("secondorder.jl")
include("differentiate_with.jl")

end # module
