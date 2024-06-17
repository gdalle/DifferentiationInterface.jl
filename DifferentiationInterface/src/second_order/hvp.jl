## Docstrings

"""
    prepare_hvp(f, backend, x, dx) -> extras

Create an `extras` object that can be given to [`hvp`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp end

function prepare_hvp_batched end

"""
    prepare_hvp_same_point(f, backend, x, dx) -> extras_same

Create an `extras_same` object that can be given to [`hvp`](@ref) and its variants _if they are applied at the same point `x`_.

!!! warning
    If the function or the point changes in any way, the result of preparation will be invalidated, and you will need to run it again.
"""
function prepare_hvp_same_point end

function prepare_hvp_batched_same_point end

"""
    hvp(f, backend, x, dx, [extras]) -> dg

Compute the Hessian-vector product of `f` at point `x` with seed `dx`.
"""
function hvp end

function hvp_batched end

"""
    hvp!(f, dg, backend, x, dx, [extras]) -> dg

Compute the Hessian-vector product of `f` at point `x` with seed `dx`, overwriting `dg`.
"""
function hvp! end

function hvp_batched! end

## Preparation

### Extras types

"""
    HVPExtras

Abstract type for additional information needed by [`hvp`](@ref) and its variants.
"""
abstract type HVPExtras <: Extras end

struct NoHVPExtras <: HVPExtras end

struct InnerGradient{F,B}
    f::F
    backend::B
end

function (ig::InnerGradient)(x)
    @compat (; f, backend) = ig
    return gradient(f, backend, x)
end

struct InnerPushforwardFixedSeed{F,B,DX}
    f::F
    backend::B
    dx::DX
end

function (ipfs::InnerPushforwardFixedSeed)(x)
    @compat (; f, backend, dx) = ipfs
    return pushforward(f, backend, x, dx)
end

struct ForwardOverForwardHVPExtras{IG<:InnerGradient,E<:PushforwardExtras} <: HVPExtras
    inner_gradient::IG
    outer_pushforward_extras::E
end

struct ForwardOverReverseHVPExtras{IG<:InnerGradient,E<:PushforwardExtras} <: HVPExtras
    inner_gradient::IG
    outer_pushforward_extras::E
end

struct ReverseOverForwardHVPExtras{E<:GradientExtras} <: HVPExtras
    outer_gradient_extras::E
end

struct ReverseOverReverseHVPExtras{IG<:InnerGradient,E<:PullbackExtras} <: HVPExtras
    inner_gradient::IG
    outer_pullback_extras::E
end

### Standard

function prepare_hvp(f::F, backend::AbstractADType, x, dx) where {F}
    return prepare_hvp(f, SecondOrder(backend, backend), x, dx)
end

function prepare_hvp(f::F, backend::SecondOrder, x, dx) where {F}
    return prepare_hvp(f, backend, x, dx, hvp_mode(backend))
end

function prepare_hvp(f::F, backend::SecondOrder, x, dx, ::ForwardOverForward) where {F}
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    inner_gradient = InnerGradient(f, nested(inner(backend)))
    outer_pushforward_extras = prepare_pushforward(inner_gradient, outer(backend), x, dx)
    return ForwardOverForwardHVPExtras(inner_gradient, outer_pushforward_extras)
end

function prepare_hvp(f::F, backend::SecondOrder, x, dx, ::ForwardOverReverse) where {F}
    # pushforward of gradient
    inner_gradient = InnerGradient(f, nested(inner(backend)))
    outer_pushforward_extras = prepare_pushforward(inner_gradient, outer(backend), x, dx)
    return ForwardOverReverseHVPExtras(inner_gradient, outer_pushforward_extras)
end

function prepare_hvp(f::F, backend::SecondOrder, x, dx, ::ReverseOverForward) where {F}
    # gradient of pushforward
    # uses dx in the closure so it can't be stored
    inner_pushforward = InnerPushforwardFixedSeed(f, nested(inner(backend)), dx)
    outer_gradient_extras = prepare_gradient(inner_pushforward, outer(backend), x)
    return ReverseOverForwardHVPExtras(outer_gradient_extras)
end

function prepare_hvp(f::F, backend::SecondOrder, x, dx, ::ReverseOverReverse) where {F}
    # pullback of the gradient
    inner_gradient = InnerGradient(f, nested(inner(backend)))
    outer_pullback_extras = prepare_pullback(inner_gradient, outer(backend), x, dx)
    return ReverseOverReverseHVPExtras(inner_gradient, outer_pullback_extras)
end

### Standard, same point

function prepare_hvp_same_point(
    f::F, backend::AbstractADType, x, dx, extras::HVPExtras
) where {F}
    return extras
end

function prepare_hvp_same_point(f::F, backend::AbstractADType, x, dx) where {F}
    extras = prepare_hvp(f, backend, x, dx)
    return prepare_hvp_same_point(f, backend, x, dx, extras)
end

### Batched

function prepare_hvp_batched(f::F, backend::AbstractADType, x, dx::Batch{B}) where {F,B}
    return prepare_hvp(f, backend, x, first(dx.elements))
end

### Batched, same point

function prepare_hvp_batched_same_point(
    f::F, backend::AbstractADType, x, dx::Batch{B}, extras::HVPExtras
) where {F,B}
    return prepare_hvp_same_point(f, backend, x, first(dx.elements), extras)
end

function prepare_hvp_batched_same_point(
    f::F, backend::AbstractADType, x, dx::Batch{B}
) where {F,B}
    return prepare_hvp_same_point(f, backend, x, first(dx.elements))
end

## One argument

### Standard

function hvp(f::F, backend::AbstractADType, x, dx) where {F}
    return hvp(f, backend, x, dx, prepare_hvp(f, backend, x, dx))
end

function hvp(f::F, backend::AbstractADType, x, dx, extras::HVPExtras) where {F}
    return hvp(f, SecondOrder(backend, backend), x, dx, extras)
end

function hvp(
    f::F, backend::SecondOrder, x, dx, extras::ForwardOverForwardHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward(inner_gradient, outer(backend), x, dx, outer_pushforward_extras)
end

function hvp(
    f::F, backend::SecondOrder, x, dx, extras::ForwardOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward(inner_gradient, outer(backend), x, dx, outer_pushforward_extras)
end

function hvp(
    f::F, backend::SecondOrder, x, dx, extras::ReverseOverForwardHVPExtras
) where {F}
    @compat (; outer_gradient_extras) = extras
    inner_pushforward = InnerPushforwardFixedSeed(f, nested(inner(backend)), dx)
    return gradient(inner_pushforward, outer(backend), x, outer_gradient_extras)
end

function hvp(
    f::F, backend::SecondOrder, x, dx, extras::ReverseOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback(inner_gradient, outer(backend), x, dx, outer_pullback_extras)
end

function hvp!(f::F, dg, backend::AbstractADType, x, dx) where {F}
    return hvp!(f, dg, backend, x, dx, prepare_hvp(f, backend, x, dx))
end

function hvp!(f::F, dg, backend::AbstractADType, x, dx, extras::HVPExtras) where {F}
    return hvp!(f, dg, SecondOrder(backend, backend), x, dx, extras)
end

function hvp!(
    f::F, dg, backend::SecondOrder, x, dx, extras::ForwardOverForwardHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward!(inner_gradient, dg, outer(backend), x, dx, outer_pushforward_extras)
end

function hvp!(
    f::F, dg, backend::SecondOrder, x, dx, extras::ForwardOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward!(inner_gradient, dg, outer(backend), x, dx, outer_pushforward_extras)
end

function hvp!(
    f::F, dg, backend::SecondOrder, x, dx, extras::ReverseOverForwardHVPExtras
) where {F}
    @compat (; outer_gradient_extras) = extras
    inner_pushforward = InnerPushforwardFixedSeed(f, nested(inner(backend)), dx)
    return gradient!(inner_pushforward, dg, outer(backend), x, outer_gradient_extras)
end

function hvp!(
    f::F, dg, backend::SecondOrder, x, dx, extras::ReverseOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback!(inner_gradient, dg, outer(backend), x, dx, outer_pullback_extras)
end

### Batched

function hvp_batched(f::F, backend::AbstractADType, x, dx, extras::HVPExtras) where {F}
    return hvp_batched(f, SecondOrder(backend, backend), x, dx, extras)
end

function hvp_batched(
    f::F, backend::SecondOrder, x, dx::Batch{B}, extras::HVPExtras
) where {F,B}
    dg_elements = ntuple(Val{B}()) do l
        hvp(f, backend, x, dx.elements[l], extras)
    end
    return Batch(dg_elements)
end

function hvp_batched!(f::F, dg, backend::AbstractADType, x, dx, extras::HVPExtras) where {F}
    return hvp_batched!(f, dg, SecondOrder(backend, backend), x, dx, extras)
end

function hvp_batched!(
    f::F, dg::Batch{B}, backend::SecondOrder, x, dx::Batch{B}, extras::HVPExtras
) where {F,B}
    for l in 1:B
        hvp!(f, dg.elements[l], backend, x, dx.elements[l], extras)
    end
end
