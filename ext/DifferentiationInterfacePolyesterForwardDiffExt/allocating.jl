
## Pushforward

function DI.value_and_pushforward(
    f, ::AutoPolyesterForwardDiff{C}, x, dx, extras::Nothing
) where {C}
    return DI.value_and_pushforward(f, AutoForwardDiff{C,Nothing}(nothing), x, dx, extras)
end

function DI.value_and_pushforward!!(
    f, dy, ::AutoPolyesterForwardDiff{C}, x, dx, extras::Nothing
) where {C}
    return DI.value_and_pushforward!!(
        f, dy, AutoForwardDiff{C,Nothing}(nothing), x, dx, extras
    )
end

## Gradient

function DI.value_and_gradient!!(
    f,
    grad::AbstractVector,
    ::AutoPolyesterForwardDiff{C},
    x::AbstractVector,
    extras::Nothing,
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return f(x), grad
end

function DI.gradient!!(
    f,
    grad::AbstractVector,
    ::AutoPolyesterForwardDiff{C},
    x::AbstractVector,
    extras::Nothing,
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return grad
end

function DI.value_and_gradient(
    f, backend::AutoPolyesterForwardDiff, x::AbstractVector, extras::Nothing
)
    return DI.value_and_gradient!!(f, similar(x), backend, x, extras)
end

function DI.gradient(
    f, backend::AutoPolyesterForwardDiff, x::AbstractVector, extras::Nothing
)
    return DI.gradient!!(f, similar(x), backend, x, extras)
end

## Jacobian

function DI.value_and_jacobian!!(
    f, jac::AbstractMatrix, ::AutoPolyesterForwardDiff{C}, x::AbstractArray, extras::Nothing
) where {C}
    return f(x), threaded_jacobian!(f, jac, x, Chunk{C}())
end

function DI.jacobian!!(
    f, jac::AbstractMatrix, ::AutoPolyesterForwardDiff{C}, x::AbstractArray, extras::Nothing
) where {C}
    return threaded_jacobian!(f, jac, x, Chunk{C}())
end

function DI.value_and_jacobian(
    f, backend::AutoPolyesterForwardDiff, x::AbstractArray, extras::Nothing
)
    y = f(x)
    return DI.value_and_jacobian!!(f, similar(y, length(y), length(x)), backend, x, extras)
end

function DI.jacobian(
    f, backend::AutoPolyesterForwardDiff, x::AbstractArray, extras::Nothing
)
    y = f(x)
    return DI.jacobian!!(f, similar(y, length(y), length(x)), backend, x, extras)
end
