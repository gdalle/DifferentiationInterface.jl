## Pushforward

function DI.prepare_pushforward(f!, y, backend::AnyAutoPolyForwardDiff, x)
    return DI.prepare_pushforward(f!, y, single_threaded(backend), x)
end

function DI.value_and_pushforward(
    f!, y, backend::AnyAutoPolyForwardDiff, x, dx, extras::PushforwardExtras
)
    return DI.value_and_pushforward(f!, y, single_threaded(backend), x, dx, extras)
end

function DI.value_and_pushforward!(
    f!, y, dy, backend::AnyAutoPolyForwardDiff, x, dx, extras::PushforwardExtras
)
    return DI.value_and_pushforward!(f!, y, dy, single_threaded(backend), x, dx, extras)
end

function DI.pushforward(
    f!, y, backend::AnyAutoPolyForwardDiff, x, dx, extras::PushforwardExtras
)
    return DI.pushforward(f!, y, single_threaded(backend), x, dx, extras)
end

function DI.pushforward!(
    f!, y, dy, backend::AnyAutoPolyForwardDiff, x, dx, extras::PushforwardExtras
)
    return DI.pushforward!(f!, y, dy, single_threaded(backend), x, dx, extras)
end

## Derivative

function DI.prepare_derivative(f!, y, backend::AnyAutoPolyForwardDiff, x)
    return DI.prepare_derivative(f!, y, single_threaded(backend), x)
end

function DI.value_and_derivative(
    f!, y, backend::AnyAutoPolyForwardDiff, x, extras::DerivativeExtras
)
    return DI.value_and_derivative(f!, y, single_threaded(backend), x, extras)
end

function DI.value_and_derivative!(
    f!, y, der, backend::AnyAutoPolyForwardDiff, x, extras::DerivativeExtras
)
    return DI.value_and_derivative!(f!, y, der, single_threaded(backend), x, extras)
end

function DI.derivative(f!, y, backend::AnyAutoPolyForwardDiff, x, extras::DerivativeExtras)
    return DI.derivative(f!, y, single_threaded(backend), x, extras)
end

function DI.derivative!(
    f!, y, der, backend::AnyAutoPolyForwardDiff, x, extras::DerivativeExtras
)
    return DI.derivative!(f!, y, der, single_threaded(backend), x, extras)
end

## Jacobian

DI.prepare_jacobian(f!, y, ::AnyAutoPolyForwardDiff, x) = NoJacobianExtras()

function DI.value_and_jacobian(
    f!, y, ::AnyAutoPolyForwardDiff{C}, x, ::NoJacobianExtras
) where {C}
    jac = similar(y, length(y), length(x))
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    f!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!, y, jac, ::AnyAutoPolyForwardDiff{C}, x, ::NoJacobianExtras
) where {C}
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    f!(y, x)
    return y, jac
end

function DI.jacobian(f!, y, ::AnyAutoPolyForwardDiff{C}, x, ::NoJacobianExtras) where {C}
    jac = similar(y, length(y), length(x))
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    return jac
end

function DI.jacobian!(
    f!, y, jac, ::AnyAutoPolyForwardDiff{C}, x, ::NoJacobianExtras
) where {C}
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    return jac
end
