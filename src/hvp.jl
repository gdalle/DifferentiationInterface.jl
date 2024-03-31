#=
Source: https://arxiv.org/abs/2403.14606 (section 8.1)

By order of preference:
- forward on reverse
- reverse on forward
- reverse on reverse
- forward on forward
=#

"""
    prepare_hvp([other_extras], f, backend, x) -> extras

Create an `extras` object that can be given to Hessian-vector product operators.
"""
function prepare_hvp(extras, f_or_f!, backend::AbstractADType, args...)
    return prepare_hvp(f_or_f!, backend, args...)
end

prepare_hvp(f, ::AbstractADType, x) = nothing

## Allocating

"""
    hvp(f, backend, x, v, [extras]) -> p
"""
function hvp(f, backend::AbstractADType, x, v, extras=prepare_hvp(f, backend, x))
    new_backend = SecondOrder(backend)
    new_extras = prepare_hvp(f, new_backend, x)
    return hvp(f, new_backend, x, v, new_extras)
end

function hvp(
    f, backend::SecondOrder, x::Number, v::Number, extras=prepare_hvp(f, backend, x)
)
    new_extras = prepare_second_derivative(extras, f, backend, x)
    return v * second_derivative(f, backend, x, new_extras)
end

function hvp(f, backend::SecondOrder, x, v, extras=prepare_hvp(f, backend, x))
    return hvp_aux(f, backend, x, v, extras, hvp_mode(backend))
end

function hvp_aux(f, backend, x, v, extras, ::ForwardOverReverse)
    # JVP of the gradient
    function gradient_closure(z)
        inner_extras = prepare_gradient(extras, f, inner(backend), z)
        return gradient(f, inner(backend), z, inner_extras)
    end
    outer_extras = prepare_pushforward(extras, gradient_closure, outer(backend), x)
    p = pushforward(gradient_closure, outer(backend), x, v, outer_extras)
    return p
end

function hvp_aux(f, backend, x, v, extras, ::ReverseOverForward)
    # gradient of the JVP
    function jvp_closure(z)
        inner_extras = prepare_pushforward(extras, f, inner(backend), z)
        return pushforward(f, inner(backend), z, v, inner_extras)
    end
    outer_extras = prepare_gradient(extras, jvp_closure, outer(backend), x)
    p = gradient(jvp_closure, outer(backend), x, outer_extras)
    return p
end

function hvp_aux(f, backend, x, v, extras, ::ReverseOverReverse)
    # VJP of the gradient
    function gradient_closure(z)
        inner_extras = prepare_gradient(extras, f, inner(backend), z)
        return gradient(f, inner(backend), z, inner_extras)
    end
    outer_extras = prepare_pullback(extras, gradient_closure, outer(backend), x)
    p = pullback(gradient_closure, outer(backend), x, v, outer_extras)
    return p
end

function hvp_aux(f, backend, x, v, extras, ::ForwardOverForward)
    # JVPs of JVPs in theory
    # also pushforward of gradient in practice
    function gradient_closure(z)
        inner_extras = prepare_gradient(extras, f, inner(backend), z)
        return gradient(f, inner(backend), z, inner_extras)
    end
    outer_extras = prepare_pushforward(extras, gradient_closure, outer(backend), x)
    p = pushforward(gradient_closure, outer(backend), x, v, outer_extras)
    return p
end

"""
    hvp!!(f, p, backend, x, v, [extras]) -> p
"""
function hvp!!(f, p, backend::AbstractADType, x, v, extras=prepare_hvp(f, backend, x))
    new_backend = SecondOrder(backend)
    new_extras = prepare_hvp(f, new_backend, x)
    return hvp!!(f, p, new_backend, x, v, new_extras)
end

function hvp!!(
    f, p, backend::SecondOrder, x::Number, v::Number, extras=prepare_hvp(f, backend, x)
)
    return v * second_derivative(f, backend, x, extras)
end

function hvp!!(f, p, backend::SecondOrder, x, v, extras=prepare_hvp(f, backend, x))
    return hvp_aux!!(f, p, backend, x, v, extras, hvp_mode(backend))
end

function hvp_aux!!(f, p, backend, x, v, extras, ::ForwardOverReverse)
    function gradient_closure(z)
        inner_extras = prepare_gradient(extras, f, inner(backend), z)
        return gradient(f, inner(backend), z, inner_extras)
    end
    outer_extras = prepare_pushforward(extras, gradient_closure, outer(backend), x)
    p = pushforward!!(gradient_closure, p, outer(backend), x, v, outer_extras)
    return p
end

function hvp_aux!!(f, p, backend, x, v, extras, ::ReverseOverForward)
    function jvp_closure(z)
        inner_extras = prepare_pushforward(extras, f, inner(backend), z)
        return pushforward(f, inner(backend), z, v, inner_extras)
    end
    outer_extras = prepare_gradient(extras, jvp_closure, outer(backend), x)
    p = gradient!!(jvp_closure, p, outer(backend), x, outer_extras)
    return p
end

function hvp_aux!!(f, p, backend, x, v, extras, ::ReverseOverReverse)
    function gradient_closure(z)
        inner_extras = prepare_gradient(extras, f, inner(backend), z)
        return gradient(f, inner(backend), z, inner_extras)
    end
    outer_extras = prepare_pullback(extras, gradient_closure, outer(backend), x)
    p = pullback!!(gradient_closure, p, outer(backend), x, v, outer_extras)
    return p
end

function hvp_aux!!(f, p, backend, x, v, extras, ::ForwardOverForward)
    function gradient_closure(z)
        inner_extras = prepare_gradient(extras, f, inner(backend), z)
        return gradient(f, inner(backend), z, inner_extras)
    end
    outer_extras = prepare_pushforward(extras, gradient_closure, outer(backend), x)
    p = pushforward!!(gradient_closure, p, outer(backend), x, v, outer_extras)
    return p
end
