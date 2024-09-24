
## Pushforward

function DI.prepare_pushforward(f, backend::AutoPolyesterForwardDiff, x, tx::Tangents)
    return DI.prepare_pushforward(f, single_threaded(backend), x, tx)
end

function DI.value_and_pushforward(
    f, prep::PushforwardPrep, backend::AutoPolyesterForwardDiff, x, tx::Tangents
)
    return DI.value_and_pushforward(f, prep, single_threaded(backend), x, tx)
end

function DI.value_and_pushforward!(
    f,
    ty::Tangents,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::Tangents,
)
    return DI.value_and_pushforward!(f, ty, prep, single_threaded(backend), x, tx)
end

function DI.pushforward(
    f, prep::PushforwardPrep, backend::AutoPolyesterForwardDiff, x, tx::Tangents
)
    return DI.pushforward(f, prep, single_threaded(backend), x, tx)
end

function DI.pushforward!(
    f,
    ty::Tangents,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::Tangents,
)
    return DI.pushforward!(f, ty, prep, single_threaded(backend), x, tx)
end

## Derivative

function DI.prepare_derivative(f, backend::AutoPolyesterForwardDiff, x)
    return DI.prepare_derivative(f, single_threaded(backend), x)
end

function DI.value_and_derivative(
    f, prep::DerivativePrep, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_and_derivative(f, prep, single_threaded(backend), x)
end

function DI.value_and_derivative!(
    f, der, prep::DerivativePrep, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_and_derivative!(f, der, prep, single_threaded(backend), x)
end

function DI.derivative(f, prep::DerivativePrep, backend::AutoPolyesterForwardDiff, x)
    return DI.derivative(f, prep, single_threaded(backend), x)
end

function DI.derivative!(f, der, prep::DerivativePrep, backend::AutoPolyesterForwardDiff, x)
    return DI.derivative!(f, der, prep, single_threaded(backend), x)
end

## Gradient

function DI.prepare_gradient(f, backend::AutoPolyesterForwardDiff, x)
    return DI.prepare_gradient(f, single_threaded(backend), x)
end

function DI.value_and_gradient!(
    f, grad, ::GradientPrep, ::AutoPolyesterForwardDiff{C}, x::AbstractVector
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return f(x), grad
end

function DI.gradient!(
    f, grad, ::GradientPrep, ::AutoPolyesterForwardDiff{C}, x::AbstractVector
) where {C}
    threaded_gradient!(f, grad, x, Chunk{C}())
    return grad
end

function DI.value_and_gradient!(
    f, grad, prep::GradientPrep, backend::AutoPolyesterForwardDiff{C}, x::AbstractArray
) where {C}
    return DI.value_and_gradient!(f, grad, prep, single_threaded(backend), x)
end

function DI.gradient!(
    f, grad, prep::GradientPrep, backend::AutoPolyesterForwardDiff{C}, x::AbstractArray
) where {C}
    return DI.gradient!(f, grad, prep, single_threaded(backend), x)
end

function DI.value_and_gradient(
    f, prep::GradientPrep, backend::AutoPolyesterForwardDiff, x::AbstractArray
)
    return DI.value_and_gradient!(f, similar(x), prep, backend, x)
end

function DI.gradient(
    f, prep::GradientPrep, backend::AutoPolyesterForwardDiff, x::AbstractArray
)
    return DI.gradient!(f, similar(x), prep, backend, x)
end

## Jacobian

DI.prepare_jacobian(f, ::AutoPolyesterForwardDiff, x) = NoJacobianPrep()

function DI.value_and_jacobian!(
    f,
    jac::AbstractMatrix,
    ::NoJacobianPrep,
    ::AutoPolyesterForwardDiff{C},
    x::AbstractArray,
) where {C}
    return f(x), threaded_jacobian!(f, jac, x, Chunk{C}())
end

function DI.jacobian!(
    f,
    jac::AbstractMatrix,
    ::NoJacobianPrep,
    ::AutoPolyesterForwardDiff{C},
    x::AbstractArray,
) where {C}
    return threaded_jacobian!(f, jac, x, Chunk{C}())
end

function DI.value_and_jacobian(
    f, prep::NoJacobianPrep, backend::AutoPolyesterForwardDiff, x::AbstractArray
)
    y = f(x)
    return DI.value_and_jacobian!(f, similar(y, length(y), length(x)), prep, backend, x)
end

function DI.jacobian(
    f, prep::NoJacobianPrep, backend::AutoPolyesterForwardDiff, x::AbstractArray
)
    y = f(x)
    return DI.jacobian!(f, similar(y, length(y), length(x)), prep, backend, x)
end

## Hessian

function DI.prepare_hessian(f, backend::AutoPolyesterForwardDiff, x)
    return DI.prepare_hessian(f, single_threaded(backend), x)
end

function DI.hessian(f, prep::HessianPrep, backend::AutoPolyesterForwardDiff, x)
    return DI.hessian(f, prep, single_threaded(backend), x)
end

function DI.hessian!(f, hess, prep::HessianPrep, backend::AutoPolyesterForwardDiff, x)
    return DI.hessian!(f, hess, prep, single_threaded(backend), x)
end

function DI.value_gradient_and_hessian(
    f, prep::HessianPrep, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_gradient_and_hessian(f, prep, single_threaded(backend), x)
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, prep::HessianPrep, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_gradient_and_hessian!(f, grad, hess, prep, single_threaded(backend), x)
end

## Second derivative

function DI.prepare_second_derivative(f, backend::AutoPolyesterForwardDiff, x)
    return DI.prepare_second_derivative(f, single_threaded(backend), x)
end

function DI.value_derivative_and_second_derivative(
    f, prep::SecondDerivativePrep, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_derivative_and_second_derivative(f, prep, single_threaded(backend), x)
end

function DI.value_derivative_and_second_derivative!(
    f, der, der2, prep::SecondDerivativePrep, backend::AutoPolyesterForwardDiff, x
)
    return DI.value_derivative_and_second_derivative!(
        f, der, der2, prep, single_threaded(backend), x
    )
end

function DI.second_derivative(
    f, prep::SecondDerivativePrep, backend::AutoPolyesterForwardDiff, x
)
    return DI.second_derivative(f, prep, single_threaded(backend), x)
end

function DI.second_derivative!(
    f, der2, prep::SecondDerivativePrep, backend::AutoPolyesterForwardDiff, x
)
    return DI.second_derivative!(f, der2, prep, single_threaded(backend), x)
end
