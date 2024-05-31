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
    make_zero

struct AutoDeferredEnzyme{M} <: ADTypes.AbstractADType
    mode::M
end

ADTypes.mode(backend::AutoDeferredEnzyme) = ADTypes.mode(AutoEnzyme(backend.mode))

DI.backend_package_name(::AutoDeferredEnzyme) = "DeferredEnzyme"

DI.nested(backend::AutoEnzyme) = AutoDeferredEnzyme(backend.mode)

const AnyAutoEnzyme{M} = Union{AutoEnzyme{M},AutoDeferredEnzyme{M}}

# forward mode if possible
forward_mode(backend::AnyAutoEnzyme{<:Mode}) = backend.mode
forward_mode(::AnyAutoEnzyme{Nothing}) = Forward

# reverse mode if possible
reverse_mode(backend::AnyAutoEnzyme{<:Mode}) = backend.mode
reverse_mode(::AnyAutoEnzyme{Nothing}) = Reverse

DI.check_available(::AutoEnzyme) = true

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DI.basis(::AutoEnzyme, a::AbstractArray{T}, i::CartesianIndex) where {T}
    b = zero(a)
    b[i] = one(T)
    return b
end

function (spig::DI.SelfPreparingInnerGradient{F,<:AnyAutoEnzyme})(x) where {F}
    @compat (; f, backend) = spig
    return DI.gradient(f, backend, x)
end

function (spid::DI.SelfPreparingInnerDerivative{F,<:AnyAutoEnzyme})(x) where {F}
    @compat (; f, backend) = spid
    return DI.derivative(f, backend, x)
end

function (spipfs::DI.SelfPreparingInnerPushforwardFixedSeed{F,<:AnyAutoEnzyme})(x) where {F}
    @compat (; f, backend, v) = spipfs
    return DI.pushforward(f, backend, x, v)
end

include("forward_onearg.jl")
include("forward_twoarg.jl")

include("reverse_onearg.jl")
include("reverse_twoarg.jl")

end # module
