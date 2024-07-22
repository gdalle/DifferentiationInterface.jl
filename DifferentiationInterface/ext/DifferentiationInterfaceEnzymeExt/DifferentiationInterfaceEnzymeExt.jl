module DifferentiationInterfaceEnzymeExt

using ADTypes: ADTypes, AutoEnzyme
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    JacobianExtras,
    PullbackExtras,
    PushforwardExtras,
    NoDerivativeExtras,
    NoGradientExtras,
    NoJacobianExtras,
    NoPullbackExtras,
    NoPushforwardExtras,
    pick_batchsize
using DocStringExtensions
using Enzyme:
    Active,
    Const,
    Duplicated,
    DuplicatedNoNeed,
    Forward,
    ForwardMode,
    Mode,
    Reverse,
    ReverseWithPrimal,
    ReverseSplitWithPrimal,
    ReverseMode,
    autodiff,
    autodiff_deferred,
    autodiff_deferred_thunk,
    autodiff_thunk,
    chunkedonehot,
    gradient,
    gradient!,
    jacobian,
    make_zero,
    make_zero!

const CONSTANT_FUNCTION_ERROR = """`AutoEnzyme(constant_function=false)` is not yet supported by DifferentiationInterface. For the time being, please use `AutoEnzyme(constant_function=true)` and avoid closures containing differentiable data."""

struct AutoDeferredEnzyme{M,constant_function} <: ADTypes.AbstractADType
    mode::M
end

ADTypes.mode(backend::AutoDeferredEnzyme) = ADTypes.mode(AutoEnzyme(backend.mode))

function DI.nested(backend::AutoEnzyme{M,constant_function}) where {M}
    return AutoDeferredEnzyme{M,constant_function}(backend.mode)
end

const AnyAutoEnzyme{M,constant_function} = Union{
    AutoEnzyme{M,constant_function},AutoDeferredEnzyme{M,constant_function}
}

# forward mode if possible
forward_mode(backend::AnyAutoEnzyme{<:Mode}) = backend.mode
forward_mode(::AnyAutoEnzyme{Nothing}) = Forward

# reverse mode if possible
reverse_mode(backend::AnyAutoEnzyme{<:Mode}) = backend.mode
reverse_mode(::AnyAutoEnzyme{Nothing}) = Reverse

DI.check_available(::AutoEnzyme) = true

# until https://github.com/EnzymeAD/Enzyme.jl/pull/1545 is merged
DI.pick_batchsize(::AnyAutoEnzyme, dimension::Integer) = min(dimension, 16)

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DI.basis(::AutoEnzyme, a::AbstractArray{T}, i::CartesianIndex) where {T}
    b = zero(a)
    b[i] = one(T)
    return b
end

include("forward_onearg.jl")
include("forward_twoarg.jl")

include("reverse_onearg.jl")
include("reverse_twoarg.jl")

end # module
