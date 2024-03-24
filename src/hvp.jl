#=
Source: https://arxiv.org/abs/2403.14606 (section 8.1)

By order of preference:
- forward on reverse
- reverse on forward
- reverse on reverse
- forward on forward
=#

## Allocating

"""
    hvp(f, backend, x, v, [extras]) -> p
"""
function hvp(
    f::F, backend::AbstractADType, x::Number, v, extras=prepare_hvp(f, backend, x)
) where {F}
    return v * second_derivative(f, backend, x, extras)
end

function hvp(
    f::F, backend::AbstractADType, x, v, extras=prepare_hvp(f, backend, x)
) where {F}
    new_backend = SecondOrder(backend, backend)
    new_extras = prepare_hvp(f, new_backend, x)
    return hvp(f, new_backend, x, v, new_extras)
end

function hvp(f::F, backend::SecondOrder, x, v, extras=prepare_hvp(backend, f, x)) where {F}
    return hvp_aux(f, backend, x, v, extras, hvp_mode(backend))
end

function hvp_aux(f::F, backend, x, v, extras, ::ForwardOverReverse) where {F}
    # JVP of the gradient
    gradient_closure(z) = gradient(f, inner(backend), z, inner(extras))
    p = pushforward(gradient_closure, outer(backend), x, v, outer(extras))
    return p
end

function hvp_aux(f::F, backend, x, v, extras, ::ReverseOverForward) where {F}
    # gradient of the JVP
    jvp_closure(z) = pushforward(f, inner(backend), z, v, inner(extras))
    p = gradient(jvp_closure, outer(backend), x, outer(extras))
    return p
end

function hvp_aux(f::F, backend, x, v, extras, ::ReverseOverReverse) where {F}
    # VJP of the gradient
    gradient_closure(z) = gradient(f, inner(backend), z, inner(extras))
    p = pullback(gradient_closure, outer(backend), x, v, outer(extras))
    return p
end

function hvp_aux(f::F, backend, x, v, extras, ::ForwardOverForward) where {F}
    # JVPs of JVPs in theory
    # also pushforward of gradient in practice
    gradient_closure(z) = gradient(f, inner(backend), z, inner(extras))
    p = pushforward(gradient_closure, outer(backend), x, v, outer(extras))
    return p
end
