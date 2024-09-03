
## Pushforward

function DI.prepare_pushforward(f, backend::AutoPolyesterForwardDiff, x, tx::Tangents)
    return DI.prepare_pushforward(f, single_threaded(backend), x, tx)
end

function DI.value_and_pushforward(
    f, extras::PushforwardExtras, backend::AutoPolyesterForwardDiff, x, tx::Tangents
)
    return DI.value_and_pushforward(f, extras, single_threaded(backend), x, tx)
end

function DI.value_and_pushforward!(
    f,
    ty::Tangents,
    extras::PushforwardExtras,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::Tangents,
)
    return DI.value_and_pushforward!(f, ty, extras, single_threaded(backend), x, tx)
end

function DI.pushforward(
    f, extras::PushforwardExtras, backend::AutoPolyesterForwardDiff, x, tx::Tangents
)
    return DI.pushforward(f, extras, single_threaded(backend), x, tx)
end

function DI.pushforward!(
    f,
    ty::Tangents,
    extras::PushforwardExtras,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::Tangents,
)
    return DI.pushforward!(f, ty, extras, single_threaded(backend), x, tx)
end

## Derivative

function DI.prepare_derivative(f, backend::AutoPolyesterForwardDiff, x)
    return DI.prepare_derivative(f, single_threaded(backend), x)
end

function DI.value_and_derivative(
    f, extras::PushforwardDerivativeExtras, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_and_derivative(f, extras, single_threaded(backend), x)
end

function DI.value_and_derivative!(
    f, der, extras::PushforwardDerivativeExtras, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_and_derivative!(f, der, extras, single_threaded(backend), x)
end

function DI.derivative(
    f, extras::PushforwardDerivativeExtras, backend::AutoPolyesterForwardDiff, x
)
    return DI.derivative(f, extras, single_threaded(backend), x)
end

function DI.derivative!(
    f, der, extras::PushforwardDerivativeExtras, backend::AutoPolyesterForwardDiff, x
)
    return DI.derivative!(f, der, extras, single_threaded(backend), x)
end

## Gradient

function DI.prepare_gradient(f, backend::AutoPolyesterForwardDiff, x)
    return DI.prepare_gradient(f, single_threaded(backend), x)
end

function DI.value_and_gradient!(
    f, grad, ::GradientExtras, ::AutoPolyesterForwardDiff{C}, x::AbstractVector
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return f(x), grad
end

function DI.gradient!(
    f, grad, ::GradientExtras, ::AutoPolyesterForwardDiff{C}, x::AbstractVector
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return grad
end

function DI.value_and_gradient!(
    f, grad, extras::GradientExtras, backend::AutoPolyesterForwardDiff{C}, x::AbstractArray
) where {C}
    return DI.value_and_gradient!(f, grad, extras, single_threaded(backend), x)
end

function DI.gradient!(
    f, grad, extras::GradientExtras, backend::AutoPolyesterForwardDiff{C}, x::AbstractArray
) where {C}
    return DI.gradient!(f, grad, extras, single_threaded(backend), x)
end

function DI.value_and_gradient(
    f, extras::GradientExtras, backend::AutoPolyesterForwardDiff, x::AbstractArray
)
    return DI.value_and_gradient!(f, similar(x), extras, backend, x)
end

function DI.gradient(
    f, extras::GradientExtras, backend::AutoPolyesterForwardDiff, x::AbstractArray
)
    return DI.gradient!(f, similar(x), extras, backend, x)
end

## Jacobian

DI.prepare_jacobian(f, ::AutoPolyesterForwardDiff, x) = NoJacobianExtras()

function DI.value_and_jacobian!(
    f,
    jac::AbstractMatrix,
    ::NoJacobianExtras,
    ::AutoPolyesterForwardDiff{C},
    x::AbstractArray,
) where {C}
    return f(x), threaded_jacobian!(f, jac, x, Chunk{C}())
end

function DI.jacobian!(
    f,
    jac::AbstractMatrix,
    ::NoJacobianExtras,
    ::AutoPolyesterForwardDiff{C},
    x::AbstractArray,
) where {C}
    return threaded_jacobian!(f, jac, x, Chunk{C}())
end

function DI.value_and_jacobian(
    f, extras::NoJacobianExtras, backend::AutoPolyesterForwardDiff, x::AbstractArray
)
    y = f(x)
    return DI.value_and_jacobian!(f, similar(y, length(y), length(x)), extras, backend, x)
end

function DI.jacobian(
    f, extras::NoJacobianExtras, backend::AutoPolyesterForwardDiff, x::AbstractArray
)
    y = f(x)
    return DI.jacobian!(f, similar(y, length(y), length(x)), extras, backend, x)
end

## Hessian

function DI.prepare_hessian(f, backend::AutoPolyesterForwardDiff, x)
    return DI.prepare_hessian(f, single_threaded(backend), x)
end

function DI.hessian(f, extras::HessianExtras, backend::AutoPolyesterForwardDiff, x)
    return DI.hessian(f, extras, single_threaded(backend), x)
end

function DI.hessian!(f, hess, extras::HessianExtras, backend::AutoPolyesterForwardDiff, x)
    return DI.hessian!(f, hess, extras, single_threaded(backend), x)
end

function DI.value_gradient_and_hessian(
    f, extras::HessianExtras, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_gradient_and_hessian(f, extras, single_threaded(backend), x)
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, extras::HessianExtras, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_gradient_and_hessian!(
        f, grad, hess, extras, single_threaded(backend), x
    )
end
