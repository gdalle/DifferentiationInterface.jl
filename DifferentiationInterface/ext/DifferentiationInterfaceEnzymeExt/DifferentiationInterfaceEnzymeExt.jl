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
    pick_chunksize
using DocStringExtensions
using Enzyme:
    Active,
    Const,
    Duplicated,
    DuplicatedNoNeed,
    Forward,
    ForwardMode,
    Reverse,
    ReverseWithPrimal,
    ReverseSplitWithPrimal,
    ReverseMode,
    autodiff,
    autodiff_thunk,
    chunkedonehot,
    gradient,
    gradient!,
    jacobian,
    make_zero
using SimpleUnPack: @unpack

const AutoForwardEnzyme = AutoEnzyme{<:ForwardMode}
const AutoForwardOrNothingEnzyme = Union{AutoEnzyme{<:ForwardMode},AutoEnzyme{Nothing}}
const AutoReverseEnzyme = AutoEnzyme{<:ReverseMode}
const AutoReverseOrNothingEnzyme = Union{AutoEnzyme{<:ReverseMode},AutoEnzyme{Nothing}}

forward_mode(backend::AutoEnzyme{<:ForwardMode}) = backend.mode
forward_mode(::AutoEnzyme{Nothing}) = Forward

reverse_mode(backend::AutoEnzyme{<:ReverseMode}) = backend.mode
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

include("forward_onearg.jl")
include("forward_twoarg.jl")

include("reverse_onearg.jl")
include("reverse_twoarg.jl")

end # module
