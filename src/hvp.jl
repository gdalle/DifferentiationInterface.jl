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

"""
    prepare_hvp([other_extras], f, backend, x) -> extras

Create an `extras` object subtyping [`HVPExtras`](@ref) that can be given to Hessian-vector product operators.
"""
function prepare_hvp(::Extras, f_or_f!, backend::AbstractADType, args...)
    return prepare_hvp(f_or_f!, backend, args...)
end

prepare_hvp(f, ::AbstractADType, x) = NoHVPExtras()

## Allocating

"""
    hvp(f, backend, x, v, [extras]) -> p
"""
function hvp(f, backend::AbstractADType, x, v, extras::HVPExtras=prepare_hvp(f, backend, x))
    new_backend = SecondOrder(backend)
    new_extras = prepare_hvp(f, new_backend, x)
    return hvp(f, new_backend, x, v, new_extras)
end

function hvp(f, backend::SecondOrder, x, v, extras::HVPExtras=prepare_hvp(f, backend, x))
    return hvp_aux(f, backend, x, v, extras, hvp_mode(backend))
end

function hvp_aux(f, backend, x, v, extras, ::ForwardOverReverse)
    # JVP of the gradient
    gradient_closure(z) = gradient(f, inner(backend), z)
    p = pushforward(gradient_closure, outer(backend), x, v)
    return p
end

function hvp_aux(f, backend, x, v, extras, ::ReverseOverForward)
    # gradient of the JVP
    pushforward_closure(z) = pushforward(f, inner(backend), z, v)
    p = gradient(pushforward_closure, outer(backend), x)
    return p
end

function hvp_aux(f, backend, x, v, extras, ::ReverseOverReverse)
    # VJP of the gradient
    gradient_closure(z) = gradient(f, inner(backend), z)
    p = pullback(gradient_closure, outer(backend), x, v)
    return p
end

function hvp_aux(f, backend, x, v, extras, ::ForwardOverForward)
    # JVPs of JVPs in theory
    # also pushforward of gradient in practice
    gradient_closure(z) = gradient(f, inner(backend), z)
    p = pushforward(gradient_closure, outer(backend), x, v)
    return p
end

"""
    hvp!!(f, p, backend, x, v, [extras]) -> p
"""
function hvp!!(
    f, p, backend::AbstractADType, x, v, extras::HVPExtras=prepare_hvp(f, backend, x)
)
    new_backend = SecondOrder(backend)
    new_extras = prepare_hvp(f, new_backend, x)
    return hvp!!(f, p, new_backend, x, v, new_extras)
end

function hvp!!(
    f, p, backend::SecondOrder, x, v, extras::HVPExtras=prepare_hvp(f, backend, x)
)
    return hvp_aux!!(f, p, backend, x, v, extras, hvp_mode(backend))
end

function hvp_aux!!(f, p, backend, x, v, extras, ::ForwardOverReverse)
    gradient_closure(z) = gradient(f, inner(backend), z)
    p = pushforward!!(gradient_closure, p, outer(backend), x, v)
    return p
end

function hvp_aux!!(f, p, backend, x, v, extras, ::ReverseOverForward)
    pushforward_closure(z) = pushforward(f, inner(backend), z, v)
    p = gradient!!(pushforward_closure, p, outer(backend), x)
    return p
end

function hvp_aux!!(f, p, backend, x, v, extras, ::ReverseOverReverse)
    gradient_closure(z) = gradient(f, inner(backend), z)
    p = pullback!!(gradient_closure, p, outer(backend), x, v)
    return p
end

function hvp_aux!!(f, p, backend, x, v, extras, ::ForwardOverForward)
    gradient_closure(z) = gradient(f, inner(backend), z)
    p = pushforward!!(gradient_closure, p, outer(backend), x, v)
    return p
end
