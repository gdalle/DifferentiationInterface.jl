## Pushforward

function DI.prepare_pushforward(f!, backend::AnyAutoPolyForwardDiff, y, x)
    return DI.prepare_pushforward(f!, single_threaded(backend), y, x)
end

function DI.value_and_pushforward!!(
    f!, y, dy, backend::AnyAutoPolyForwardDiff, x, dx, extras::PushforwardExtras
)
    return DI.value_and_pushforward!!(f!, y, dy, single_threaded(backend), x, dx, extras)
end

## Derivative

function DI.prepare_derivative(f!, backend::AnyAutoPolyForwardDiff, y, x)
    return DI.prepare_derivative(f!, single_threaded(backend), y, x)
end

function DI.value_and_derivative!!(
    f!, y, der, backend::AnyAutoPolyForwardDiff, x, extras::DerivativeExtras
)
    return DI.value_and_derivative!!(f!, y, der, single_threaded(backend), x, extras)
end

## Jacobian

DI.prepare_jacobian(f!, ::AnyAutoPolyForwardDiff, y, x) = NoJacobianExtras()

function DI.value_and_jacobian!!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AnyAutoPolyForwardDiff{C},
    x::AbstractArray,
    ::NoJacobianExtras,
) where {C}
    f!(y, x)
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    return y, jac
end
