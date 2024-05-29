module DifferentiationInterfaceEnzymeExt

using ADTypes: ADTypes, AutoEnzyme
using Compat
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    JacobianExtras,
    PullbackExtras,
    PushforwardExtras,
    SecondDerivativeExtras,
    NoDerivativeExtras,
    NoGradientExtras,
    NoHessianExtras,
    NoHVPExtras,
    NoJacobianExtras,
    NoPullbackExtras,
    NoPushforwardExtras,
    NoSecondDerivativeExtras,
    pick_chunksize
using DocStringExtensions
using Enzyme:
    Active,
    Const,
    Mode,
    Duplicated,
    DuplicatedNoNeed,
    Forward,
    ForwardMode,
    Reverse,
    ReverseWithPrimal,
    ReverseSplitWithPrimal,
    ReverseMode,
    autodiff,
    autodiff_deferred,
    autodiff_thunk,
    chunkedonehot,
    gradient,
    gradient!,
    jacobian,
    make_zero

const AutoMixedEnzyme = AutoEnzyme{Nothing}
const AutoForwardEnzyme = AutoEnzyme{<:ForwardMode}
const AutoForwardOrNothingEnzyme = Union{AutoEnzyme{<:ForwardMode},AutoEnzyme{Nothing}}
const AutoReverseEnzyme = AutoEnzyme{<:ReverseMode}
const AutoReverseOrNothingEnzyme = Union{AutoEnzyme{<:ReverseMode},AutoEnzyme{Nothing}}

# forward mode if possible
forward_mode(backend::AutoEnzyme{<:Mode}) = backend.mode
forward_mode(::AutoEnzyme{Nothing}) = Forward

# reverse mode if possible
reverse_mode(backend::AutoEnzyme{<:Mode}) = backend.mode
reverse_mode(::AutoEnzyme{Nothing}) = Reverse

DI.check_available(::AutoEnzyme) = true

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DI.basis(::AutoEnzyme, a::AbstractArray{T}, i::CartesianIndex) where {T}
    b = zero(a)
    b[i] = one(T)
    return b
end

function zero_sametype!(x_target, x)
    x_sametype = convert(typeof(x), x_target)
    x_sametype .= zero(eltype(x_sametype))
    return x_sametype
end

include("utils.jl")

include("forward_onearg.jl")
include("forward_twoarg.jl")

include("reverse_onearg.jl")
include("reverse_twoarg.jl")

include("common_onearg.jl")

end # module
