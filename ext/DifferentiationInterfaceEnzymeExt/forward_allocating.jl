## Pushforward

function DI.value_and_pushforward(f, backend::AutoForwardEnzyme, x, dx, extras::Nothing)
    dx_sametype = convert(typeof(x), dx)
    y, new_dy = autodiff(backend.mode, f, Duplicated, Duplicated(x, dx_sametype))
    return y, new_dy
end

function DI.pushforward(f, backend::AutoForwardEnzyme, x, dx, extras::Nothing)
    dx_sametype = convert(typeof(x), dx)
    new_dy = only(autodiff(backend.mode, f, DuplicatedNoNeed, Duplicated(x, dx_sametype)))
    return new_dy
end

function DI.value_and_pushforward!!(
    f, _dy, backend::AutoForwardEnzyme, x, dx, extras::Nothing
)
    # dy cannot be passed anyway
    return DI.value_and_pushforward(f, backend, x, dx, extras)
end

function DI.pushforward!!(f, _dy, backend::AutoForwardEnzyme, x, dx, extras::Nothing)
    # dy cannot be passed anyway
    return DI.pushforward(f, backend, x, dx, extras)
end

## Gradient

function DI.gradient(f, backend::AutoForwardEnzyme, x::AbstractArray, extras::Nothing)
    return reshape(collect(gradient(backend.mode, f, x)), size(x))
end

function DI.value_and_gradient(
    f, backend::AutoForwardEnzyme, x::AbstractArray, extras::Nothing
)
    return f(x), DI.gradient(f, backend, x, extras)
end

function DI.gradient!!(
    f, _grad, backend::AutoForwardEnzyme, x::AbstractArray, extras::Nothing
)
    return DI.gradient(f, backend, x, extras)
end

function DI.value_and_gradient!!(
    f, _grad, backend::AutoForwardEnzyme, x::AbstractArray, extras::Nothing
)
    return DI.value_and_gradient(f, backend, x, extras)
end

## Jacobian

function DI.jacobian(f, backend::AutoForwardEnzyme, x::AbstractArray, extras::Nothing)
    jac_wrongshape = jacobian(backend.mode, f, x)
    nx = length(x)
    ny = length(jac_wrongshape) ÷ length(x)
    return reshape(jac_wrongshape, ny, nx)
end

function DI.value_and_jacobian(
    f, backend::AutoForwardEnzyme, x::AbstractArray, extras::Nothing
)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.jacobian!!(
    f, _jac, backend::AutoForwardEnzyme, x::AbstractArray, extras::Nothing
)
    return DI.jacobian(f, backend, x, extras)
end

function DI.value_and_jacobian!!(
    f, _jac, backend::AutoForwardEnzyme, x::AbstractArray, extras::Nothing
)
    return DI.value_and_jacobian(f, backend, x, extras)
end
