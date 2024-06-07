
## Pushforward

function DI.prepare_pushforward(f, backend::AutoPolyesterForwardDiff, x, dx)
    return DI.prepare_pushforward(f, single_threaded(backend), x, dx)
end

function DI.value_and_pushforward(
    f, backend::AutoPolyesterForwardDiff, x, dx, extras::PushforwardExtras
)
    return DI.value_and_pushforward(f, single_threaded(backend), x, dx, extras)
end

function DI.value_and_pushforward!(
    f, dy, backend::AutoPolyesterForwardDiff, x, dx, extras::PushforwardExtras
)
    return DI.value_and_pushforward!(f, dy, single_threaded(backend), x, dx, extras)
end

function DI.pushforward(
    f, backend::AutoPolyesterForwardDiff, x, dx, extras::PushforwardExtras
)
    return DI.pushforward(f, single_threaded(backend), x, dx, extras)
end

function DI.pushforward!(
    f, dy, backend::AutoPolyesterForwardDiff, x, dx, extras::PushforwardExtras
)
    return DI.pushforward!(f, dy, single_threaded(backend), x, dx, extras)
end

## Derivative

function DI.prepare_derivative(
    f, backend::AutoPolyesterForwardDiff, x
)::PushforwardDerivativeExtras
    return DI.prepare_derivative(f, single_threaded(backend), x)
end

function DI.value_and_derivative(
    f, backend::AutoPolyesterForwardDiff, x, extras::PushforwardDerivativeExtras
)
    return DI.value_and_derivative(f, single_threaded(backend), x, extras)
end

function DI.value_and_derivative!(
    f, der, backend::AutoPolyesterForwardDiff, x, extras::PushforwardDerivativeExtras
)
    return DI.value_and_derivative!(f, der, single_threaded(backend), x, extras)
end

function DI.derivative(
    f, backend::AutoPolyesterForwardDiff, x, extras::PushforwardDerivativeExtras
)
    return DI.derivative(f, single_threaded(backend), x, extras)
end

function DI.derivative!(
    f, der, backend::AutoPolyesterForwardDiff, x, extras::PushforwardDerivativeExtras
)
    return DI.derivative!(f, der, single_threaded(backend), x, extras)
end

## Gradient

function DI.prepare_gradient(f, backend::AutoPolyesterForwardDiff, x)
    return DI.prepare_gradient(f, single_threaded(backend), x)
end

function DI.value_and_gradient!(
    f, grad, ::AutoPolyesterForwardDiff{C}, x::AbstractVector, ::GradientExtras
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return f(x), grad
end

function DI.gradient!(
    f, grad, ::AutoPolyesterForwardDiff{C}, x::AbstractVector, ::GradientExtras
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return grad
end

function DI.value_and_gradient!(
    f, grad, backend::AutoPolyesterForwardDiff{C}, x::AbstractArray, extras::GradientExtras
) where {C}
    return DI.value_and_gradient!(f, grad, single_threaded(backend), x, extras)
end

function DI.gradient!(
    f, grad, backend::AutoPolyesterForwardDiff{C}, x::AbstractArray, extras::GradientExtras
) where {C}
    return DI.gradient!(f, grad, single_threaded(backend), x, extras)
end

function DI.value_and_gradient(
    f, backend::AutoPolyesterForwardDiff, x::AbstractArray, extras::GradientExtras
)
    return DI.value_and_gradient!(f, similar(x), backend, x, extras)
end

function DI.gradient(
    f, backend::AutoPolyesterForwardDiff, x::AbstractArray, extras::GradientExtras
)
    return DI.gradient!(f, similar(x), backend, x, extras)
end

## Jacobian

DI.prepare_jacobian(f, ::AutoPolyesterForwardDiff, x) = NoJacobianExtras()

function DI.value_and_jacobian!(
    f,
    jac::AbstractMatrix,
    ::AutoPolyesterForwardDiff{C},
    x::AbstractArray,
    ::NoJacobianExtras,
) where {C}
    return f(x), threaded_jacobian!(f, jac, x, Chunk{C}())
end

function DI.jacobian!(
    f,
    jac::AbstractMatrix,
    ::AutoPolyesterForwardDiff{C},
    x::AbstractArray,
    ::NoJacobianExtras,
) where {C}
    return threaded_jacobian!(f, jac, x, Chunk{C}())
end

function DI.value_and_jacobian(
    f, backend::AutoPolyesterForwardDiff, x::AbstractArray, extras::NoJacobianExtras
)
    y = f(x)
    return DI.value_and_jacobian!(f, similar(y, length(y), length(x)), backend, x, extras)
end

function DI.jacobian(
    f, backend::AutoPolyesterForwardDiff, x::AbstractArray, extras::NoJacobianExtras
)
    y = f(x)
    return DI.jacobian!(f, similar(y, length(y), length(x)), backend, x, extras)
end

## Hessian

function DI.prepare_hessian(f, backend::AutoPolyesterForwardDiff, x)
    return DI.prepare_hessian(f, single_threaded(backend), x)
end

function DI.hessian(f, backend::AutoPolyesterForwardDiff, x, extras::HessianExtras)
    return DI.hessian(f, single_threaded(backend), x, extras)
end

function DI.hessian!(f, hess, backend::AutoPolyesterForwardDiff, x, extras::HessianExtras)
    return DI.hessian!(f, hess, single_threaded(backend), x, extras)
end

function DI.value_gradient_and_hessian(
    f, backend::AutoPolyesterForwardDiff, x, extras::HessianExtras
)
    return DI.value_gradient_and_hessian(f, single_threaded(backend), x, extras)
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, backend::AutoPolyesterForwardDiff, x, extras::HessianExtras
)
    return DI.value_gradient_and_hessian!(
        f, grad, hess, single_threaded(backend), x, extras
    )
end
