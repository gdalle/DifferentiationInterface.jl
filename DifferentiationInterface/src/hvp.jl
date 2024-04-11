#=
Source: https://arxiv.org/abs/2403.14606 (section 8.1)

By order of preference:
- forward on reverse
- reverse on forward
- reverse on reverse
- forward on forward
=#

## Preparation

"""
    HVPExtras

Abstract type for additional information needed by Hessian-vector product operators.
"""
abstract type HVPExtras <: Extras end

struct NoHVPExtras <: HVPExtras end

struct ForwardOverForwardHVPExtras{C,E} <: HVPExtras
    inner_gradient_closure::C
    outer_pushforward_extras::E
end

struct ForwardOverReverseHVPExtras{C,E} <: HVPExtras
    inner_gradient_closure::C
    outer_pushforward_extras::E
end

struct ReverseOverForwardHVPExtras{C,E} <: HVPExtras
    inner_pushforward_closure_generator::C
    outer_gradient_extras::E
end

struct ReverseOverReverseHVPExtras{C,E} <: HVPExtras
    inner_gradient_closure::C
    outer_pullback_extras::E
end

"""
    prepare_hvp(f, backend, x, v) -> extras

Create an `extras` object subtyping [`HVPExtras`](@ref) that can be given to Hessian-vector product operators.

!!! warning
    Unlike the others, this preparation operator takes an additional argument `v`.
"""
prepare_hvp(f, ::AbstractADType, x, v) = NoHVPExtras()

function prepare_hvp(f, backend::SecondOrder, x, v)
    return prepare_hvp_aux(f, backend, x, v, hvp_mode(backend))
end

function prepare_hvp_aux(f, backend::SecondOrder, x, v, ::ForwardOverForward)
    # pushforward of many pushforwards in theory, but pushforward of gradient in practice
    inner_gradient_closure(z) = gradient(f, inner(backend), z)
    outer_pushforward_extras = prepare_pushforward(
        inner_gradient_closure, outer(backend), x
    )
    return ForwardOverForwardHVPExtras(inner_gradient_closure, outer_pushforward_extras)
end

function prepare_hvp_aux(f, backend::SecondOrder, x, v, ::ForwardOverReverse)
    # pushforward of gradient
    inner_gradient_closure(z) = gradient(f, inner(backend), z)
    outer_pushforward_extras = prepare_pushforward(
        inner_gradient_closure, outer(backend), x
    )
    return ForwardOverReverseHVPExtras(inner_gradient_closure, outer_pushforward_extras)
end

function prepare_hvp_aux(f, backend::SecondOrder, x, v, ::ReverseOverForward)
    # gradient of pushforward
    # uses v in the closure
    inner_pushforward_closure_generator(v) = z -> pushforward(f, inner(backend), z, v)
    outer_gradient_extras = prepare_gradient(
        inner_pushforward_closure_generator(v), outer(backend), x
    )
    return ReverseOverForwardHVPExtras(
        inner_pushforward_closure_generator, outer_gradient_extras
    )
end

function prepare_hvp_aux(f, backend::SecondOrder, x, v, ::ReverseOverReverse)
    # pullback of the gradient
    inner_gradient_closure(z) = gradient(f, inner(backend), z)
    outer_pullback_extras = prepare_pullback(inner_gradient_closure, outer(backend), x)
    return ReverseOverReverseHVPExtras(inner_gradient_closure, outer_pullback_extras)
end

## One argument

"""
    hvp(f, backend, x, v, [extras]) -> p
"""
function hvp(
    f, backend::AbstractADType, x, v, extras::HVPExtras=prepare_hvp(f, backend, x, v)
)
    new_backend = SecondOrder(backend)
    new_extras = prepare_hvp(f, new_backend, x, v)
    return hvp(f, new_backend, x, v, new_extras)
end

function hvp(f, backend::SecondOrder, x, v, extras::HVPExtras=prepare_hvp(f, backend, x, v))
    return hvp_aux(f, backend, x, v, extras)
end

function hvp_aux(f, backend, x, v, extras::ForwardOverForwardHVPExtras)
    return pushforward(
        extras.inner_gradient_closure, outer(backend), x, v, extras.outer_pushforward_extras
    )
end

function hvp_aux(f, backend, x, v, extras::ForwardOverReverseHVPExtras)
    return pushforward(
        extras.inner_gradient_closure, outer(backend), x, v, extras.outer_pushforward_extras
    )
end

function hvp_aux(f, backend, x, v, extras::ReverseOverForwardHVPExtras)
    inner_pushforward_closure = extras.inner_pushforward_closure_generator(v)
    return gradient(
        inner_pushforward_closure, outer(backend), x, extras.outer_gradient_extras
    )
end

function hvp_aux(f, backend, x, v, extras::ReverseOverReverseHVPExtras)
    return pullback(
        extras.inner_gradient_closure, outer(backend), x, v, extras.outer_pullback_extras
    )
end

"""
    hvp!(f, p, backend, x, v, [extras]) -> p
"""
function hvp!(
    f, p, backend::AbstractADType, x, v, extras::HVPExtras=prepare_hvp(f, backend, x, v)
)
    new_backend = SecondOrder(backend)
    new_extras = prepare_hvp(f, new_backend, x, v)
    return hvp!(f, p, new_backend, x, v, new_extras)
end

function hvp!(
    f, p, backend::SecondOrder, x, v, extras::HVPExtras=prepare_hvp(f, backend, x, v)
)
    return hvp_aux!(f, p, backend, x, v, extras)
end

function hvp_aux!(f, p, backend, x, v, extras::ForwardOverForwardHVPExtras)
    return pushforward!(
        extras.inner_gradient_closure,
        p,
        outer(backend),
        x,
        v,
        extras.outer_pushforward_extras,
    )
end

function hvp_aux!(f, p, backend, x, v, extras::ForwardOverReverseHVPExtras)
    return pushforward!(
        extras.inner_gradient_closure,
        p,
        outer(backend),
        x,
        v,
        extras.outer_pushforward_extras,
    )
end

function hvp_aux!(f, p, backend, x, v, extras::ReverseOverForwardHVPExtras)
    inner_pushforward_closure = extras.inner_pushforward_closure_generator(v)
    return gradient!(
        inner_pushforward_closure, p, outer(backend), x, extras.outer_gradient_extras
    )
end

function hvp_aux!(f, p, backend, x, v, extras::ReverseOverReverseHVPExtras)
    return pullback!(
        extras.inner_gradient_closure, p, outer(backend), x, v, extras.outer_pullback_extras
    )
end
