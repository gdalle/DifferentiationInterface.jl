
## Pushforward

function DI.prepare_pushforward(
    f, backend::AutoPolyesterForwardDiff, x, tx::NTuple, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_pushforward(f, single_threaded(backend), x, tx, contexts...)
end

function DI.value_and_pushforward(
    f,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_pushforward(f, prep, single_threaded(backend), x, tx, contexts...)
end

function DI.value_and_pushforward!(
    f,
    ty::NTuple,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_pushforward!(
        f, ty, prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.pushforward(
    f,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.pushforward(f, prep, single_threaded(backend), x, tx, contexts...)
end

function DI.pushforward!(
    f,
    ty::NTuple,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.pushforward!(f, ty, prep, single_threaded(backend), x, tx, contexts...)
end

## Derivative

function DI.prepare_derivative(
    f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_derivative(f, single_threaded(backend), x, contexts...)
end

function DI.value_and_derivative(
    f,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_derivative(f, prep, single_threaded(backend), x, contexts...)
end

function DI.value_and_derivative!(
    f,
    der,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_derivative!(f, der, prep, single_threaded(backend), x, contexts...)
end

function DI.derivative(
    f,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.derivative(f, prep, single_threaded(backend), x, contexts...)
end

function DI.derivative!(
    f,
    der,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.derivative!(f, der, prep, single_threaded(backend), x, contexts...)
end

## Gradient

function DI.prepare_gradient(
    f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_gradient(f, single_threaded(backend), x, contexts...)
end

function DI.value_and_gradient!(
    f,
    grad,
    ::GradientPrep,
    ::AutoPolyesterForwardDiff{K},
    x::AbstractVector,
    contexts::Vararg{Context,C},
) where {K,C}
    fc = with_contexts(f, contexts...)
    threaded_gradient!(fc, grad, x, Chunk{K}())
    return fc(x), grad
end

function DI.gradient!(
    f,
    grad,
    ::GradientPrep,
    ::AutoPolyesterForwardDiff{K},
    x::AbstractVector,
    contexts::Vararg{Context,C},
) where {K,C}
    fc = with_contexts(f, contexts...)
    threaded_gradient!(fc, grad, x, Chunk{K}())
    return grad
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::GradientPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_gradient!(f, grad, prep, single_threaded(backend), x, contexts...)
end

function DI.gradient!(
    f,
    grad,
    prep::GradientPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.gradient!(f, grad, prep, single_threaded(backend), x, contexts...)
end

function DI.value_and_gradient(
    f, prep::GradientPrep, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.value_and_gradient!(f, similar(x), prep, backend, x, contexts...)
end

function DI.gradient(
    f, prep::GradientPrep, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.gradient!(f, similar(x), prep, backend, x, contexts...)
end

## Jacobian

function DI.prepare_jacobian(
    f, ::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return NoJacobianPrep()
end

function DI.value_and_jacobian!(
    f,
    jac::AbstractMatrix,
    ::NoJacobianPrep,
    ::AutoPolyesterForwardDiff{K},
    x::AbstractArray,
    contexts::Vararg{Context,C},
) where {K,C}
    fc = with_contexts(f, contexts...)
    return fc(x), threaded_jacobian!(fc, jac, x, Chunk{K}())
end

function DI.jacobian!(
    f,
    jac::AbstractMatrix,
    ::NoJacobianPrep,
    ::AutoPolyesterForwardDiff{K},
    x::AbstractArray,
    contexts::Vararg{Context,C},
) where {K,C}
    fc = with_contexts(f, contexts...)
    return threaded_jacobian!(fc, jac, x, Chunk{K}())
end

function DI.value_and_jacobian(
    f,
    prep::NoJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x::AbstractArray,
    contexts::Vararg{Context,C},
) where {C}
    y = f(x, map(unwrap, contexts)...)
    return DI.value_and_jacobian!(
        f, similar(y, length(y), length(x)), prep, backend, x, contexts...
    )
end

function DI.jacobian(
    f,
    prep::NoJacobianPrep,
    backend::AutoPolyesterForwardDiff,
    x::AbstractArray,
    contexts::Vararg{Context,C},
) where {C}
    y = f(x, map(unwrap, contexts)...)
    return DI.jacobian!(f, similar(y, length(y), length(x)), prep, backend, x, contexts...)
end

## Hessian

function DI.prepare_hessian(
    f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_hessian(f, single_threaded(backend), x, contexts...)
end

function DI.hessian(
    f, prep::HessianPrep, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.hessian(f, prep, single_threaded(backend), x, contexts...)
end

function DI.hessian!(
    f,
    hess,
    prep::HessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.hessian!(f, hess, prep, single_threaded(backend), x, contexts...)
end

function DI.value_gradient_and_hessian(
    f, prep::HessianPrep, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.value_gradient_and_hessian(f, prep, single_threaded(backend), x, contexts...)
end

function DI.value_gradient_and_hessian!(
    f,
    grad,
    hess,
    prep::HessianPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_gradient_and_hessian!(
        f, grad, hess, prep, single_threaded(backend), x, contexts...
    )
end

## HVP

function DI.prepare_hvp(
    f, backend::AutoPolyesterForwardDiff, x, tx::NTuple, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_hvp(f, single_threaded(backend), x, tx, contexts...)
end

function DI.hvp(
    f,
    prep::HVPPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.hvp(f, prep, single_threaded(backend), x, tx, contexts...)
end

function DI.hvp!(
    f,
    tg::NTuple,
    prep::HVPPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.hvp!(f, tg, prep, single_threaded(backend), x, tx, contexts...)
end

## Second derivative

function DI.prepare_second_derivative(
    f, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_second_derivative(f, single_threaded(backend), x, contexts...)
end

function DI.value_derivative_and_second_derivative(
    f,
    prep::SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_derivative_and_second_derivative(
        f, prep, single_threaded(backend), x, contexts...
    )
end

function DI.value_derivative_and_second_derivative!(
    f,
    der,
    der2,
    prep::SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_derivative_and_second_derivative!(
        f, der, der2, prep, single_threaded(backend), x, contexts...
    )
end

function DI.second_derivative(
    f,
    prep::SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.second_derivative(f, prep, single_threaded(backend), x, contexts...)
end

function DI.second_derivative!(
    f,
    der2,
    prep::SecondDerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.second_derivative!(f, der2, prep, single_threaded(backend), x, contexts...)
end
