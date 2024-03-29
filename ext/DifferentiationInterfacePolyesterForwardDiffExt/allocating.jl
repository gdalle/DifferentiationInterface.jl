
## Pushforward

function DI.value_and_pushforward(
    f, backend::AllAutoPolyForwardDiff, x, dx, extras::Nothing
)
    return DI.value_and_pushforward(f, single_threaded(backend), x, dx, extras)
end

function DI.value_and_pushforward!!(
    f, dy, backend::AllAutoPolyForwardDiff, x, dx, extras::Nothing
)
    return DI.value_and_pushforward!!(f, dy, single_threaded(backend), x, dx, extras)
end

function DI.pushforward(f, backend::AllAutoPolyForwardDiff, x, dx, extras::Nothing)
    return DI.pushforward(f, single_threaded(backend), x, dx, extras)
end

function DI.pushforward!!(f, dy, backend::AllAutoPolyForwardDiff, x, dx, extras::Nothing)
    return DI.pushforward!!(f, dy, single_threaded(backend), x, dx, extras)
end

## Derivative

function DI.value_and_derivative(f, backend::AllAutoPolyForwardDiff, x, extras::Nothing)
    return DI.value_and_derivative(f, single_threaded(backend), x, extras)
end

function DI.value_and_derivative!!(
    f, dy, backend::AllAutoPolyForwardDiff, x, extras::Nothing
)
    return DI.value_and_derivative!!(f, dy, single_threaded(backend), x, extras)
end

function DI.derivative(f, backend::AllAutoPolyForwardDiff, x, extras::Nothing)
    return DI.derivative(f, single_threaded(backend), x, extras)
end

function DI.derivative!!(f, dy, backend::AllAutoPolyForwardDiff, x, extras::Nothing)
    return DI.derivative!!(f, dy, single_threaded(backend), x, extras)
end

## Gradient

function DI.value_and_gradient!!(
    f, grad::AbstractVector, ::AllAutoPolyForwardDiff{C}, x::AbstractVector, extras::Nothing
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return f(x), grad
end

function DI.gradient!!(
    f, grad::AbstractVector, ::AllAutoPolyForwardDiff{C}, x::AbstractVector, extras::Nothing
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return grad
end

function DI.value_and_gradient(
    f, backend::AllAutoPolyForwardDiff, x::AbstractVector, extras::Nothing
)
    return DI.value_and_gradient!!(f, similar(x), backend, x, extras)
end

function DI.gradient(f, backend::AllAutoPolyForwardDiff, x::AbstractVector, extras::Nothing)
    return DI.gradient!!(f, similar(x), backend, x, extras)
end

## Jacobian

function DI.value_and_jacobian!!(
    f, jac::AbstractMatrix, ::AllAutoPolyForwardDiff{C}, x::AbstractArray, extras::Nothing
) where {C}
    return f(x), threaded_jacobian!(f, jac, x, Chunk{C}())
end

function DI.jacobian!!(
    f, jac::AbstractMatrix, ::AllAutoPolyForwardDiff{C}, x::AbstractArray, extras::Nothing
) where {C}
    return threaded_jacobian!(f, jac, x, Chunk{C}())
end

function DI.value_and_jacobian(
    f, backend::AllAutoPolyForwardDiff, x::AbstractArray, extras::Nothing
)
    y = f(x)
    return DI.value_and_jacobian!!(f, similar(y, length(y), length(x)), backend, x, extras)
end

function DI.jacobian(f, backend::AllAutoPolyForwardDiff, x::AbstractArray, extras::Nothing)
    y = f(x)
    return DI.jacobian!!(f, similar(y, length(y), length(x)), backend, x, extras)
end

## Hessian

function DI.hessian(f, backend::AllAutoPolyForwardDiff, x, extras::Nothing)
    return DI.hessian(f, single_threaded(backend), x, extras)
end

function DI.hessian!!(f, dy, backend::AllAutoPolyForwardDiff, x, extras::Nothing)
    return DI.hessian!!(f, dy, single_threaded(backend), x, extras)
end
