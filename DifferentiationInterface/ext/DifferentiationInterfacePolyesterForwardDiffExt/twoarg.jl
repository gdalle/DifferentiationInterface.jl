## Pushforward

function DI.prepare_pushforward(f!, y, backend::AutoPolyesterForwardDiff, x, tx::Tangents)
    return DI.prepare_pushforward(f!, y, single_threaded(backend), x, tx)
end

function DI.value_and_pushforward(
    f!, y, extras::PushforwardExtras, backend::AutoPolyesterForwardDiff, x, tx::Tangents
)
    return DI.value_and_pushforward(f!, y, extras, single_threaded(backend), x, tx)
end

function DI.value_and_pushforward!(
    f!,
    y,
    ty::Tangents,
    extras::PushforwardExtras,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::Tangents,
)
    return DI.value_and_pushforward!(f!, y, ty, extras, single_threaded(backend), x, tx)
end

function DI.pushforward(
    f!, y, extras::PushforwardExtras, backend::AutoPolyesterForwardDiff, x, tx::Tangents
)
    return DI.pushforward(f!, y, extras, single_threaded(backend), x, tx)
end

function DI.pushforward!(
    f!,
    y,
    ty::Tangents,
    extras::PushforwardExtras,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::Tangents,
)
    return DI.pushforward!(f!, y, ty, extras, single_threaded(backend), x, tx)
end

## Derivative

function DI.prepare_derivative(f!, y, backend::AutoPolyesterForwardDiff, x)
    return DI.prepare_derivative(f!, y, single_threaded(backend), x)
end

function DI.value_and_derivative(
    f!, y, extras::DerivativeExtras, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_and_derivative(f!, y, extras, single_threaded(backend), x)
end

function DI.value_and_derivative!(
    f!, y, der, extras::DerivativeExtras, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_and_derivative!(f!, y, der, extras, single_threaded(backend), x)
end

function DI.derivative(
    f!, y, extras::DerivativeExtras, backend::AutoPolyesterForwardDiff, x
)
    return DI.derivative(f!, y, extras, single_threaded(backend), x)
end

function DI.derivative!(
    f!, y, der, extras::DerivativeExtras, backend::AutoPolyesterForwardDiff, x
)
    return DI.derivative!(f!, y, der, extras, single_threaded(backend), x)
end

## Jacobian

DI.prepare_jacobian(f!, y, ::AutoPolyesterForwardDiff, x) = NoJacobianExtras()

function DI.value_and_jacobian(
    f!, y, ::NoJacobianExtras, ::AutoPolyesterForwardDiff{C}, x
) where {C}
    jac = similar(y, length(y), length(x))
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    f!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!, y, jac, ::NoJacobianExtras, ::AutoPolyesterForwardDiff{C}, x
) where {C}
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    f!(y, x)
    return y, jac
end

function DI.jacobian(f!, y, ::NoJacobianExtras, ::AutoPolyesterForwardDiff{C}, x) where {C}
    jac = similar(y, length(y), length(x))
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    return jac
end

function DI.jacobian!(
    f!, y, jac, ::NoJacobianExtras, ::AutoPolyesterForwardDiff{C}, x
) where {C}
    threaded_jacobian!(f!, y, jac, x, Chunk{C}())
    return jac
end
