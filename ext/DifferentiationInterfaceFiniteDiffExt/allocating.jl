## Pushforward

DI.prepare_pushforward(f, ::AnyAutoFiniteDiff, x) = NoPushforwardExtras()

function DI.pushforward(f, backend::AnyAutoFiniteDiff, x, dx, ::NoPushforwardExtras)
    step(t::Number) = f(x .+ t .* dx)
    new_dy = finite_difference_derivative(step, zero(eltype(x)), fdtype(backend))
    return new_dy
end

function DI.value_and_pushforward(
    f, backend::AnyAutoFiniteDiff, x, dx, ::NoPushforwardExtras
)
    y = f(x)
    step(t::Number) = f(x .+ t .* dx)
    new_dy = finite_difference_derivative(
        step, zero(eltype(x)), fdtype(backend), eltype(y), y
    )
    return y, new_dy
end

## Derivative

DI.prepare_derivative(f, ::AnyAutoFiniteDiff, x) = NoDerivativeExtras()

function DI.derivative(f, backend::AnyAutoFiniteDiff, x, ::NoDerivativeExtras)
    return finite_difference_derivative(f, x, fdtype(backend))
end

function DI.value_and_derivative(f, backend::AnyAutoFiniteDiff, x, ::NoDerivativeExtras)
    y = f(x)
    return y, finite_difference_derivative(f, x, fdtype(backend), eltype(y), y)
end

function DI.derivative!!(f, der, backend::AnyAutoFiniteDiff, x, extras::NoDerivativeExtras)
    return DI.derivative(f, backend, x, extras)
end

function DI.value_and_derivative!!(
    f, der, backend::AnyAutoFiniteDiff, x, extras::NoDerivativeExtras
)
    return DI.value_and_derivative(f, backend, x, extras)
end

## Gradient

DI.prepare_gradient(f, ::AnyAutoFiniteDiff, x) = NoGradientExtras()

function DI.gradient(f, backend::AnyAutoFiniteDiff, x::AbstractArray, ::NoGradientExtras)
    return finite_difference_gradient(f, x, fdtype(backend))
end

function DI.value_and_gradient(
    f, backend::AnyAutoFiniteDiff, x::AbstractArray, ::NoGradientExtras
)
    y = f(x)
    return y, finite_difference_gradient(f, x, fdtype(backend), typeof(y), y)
end

function DI.gradient!!(
    f, grad, backend::AnyAutoFiniteDiff, x::AbstractArray, ::NoGradientExtras
)
    return finite_difference_gradient!(grad, f, x, fdtype(backend))
end

function DI.value_and_gradient!!(
    f, grad, backend::AnyAutoFiniteDiff, x::AbstractArray, ::NoGradientExtras
)
    y = f(x)
    return y, finite_difference_gradient!(grad, f, x, fdtype(backend), typeof(y), y)
end

## Jacobian

DI.prepare_jacobian(f, ::AnyAutoFiniteDiff, x) = NoJacobianExtras()

function DI.jacobian(f, backend::AnyAutoFiniteDiff, x, ::NoJacobianExtras)
    return finite_difference_jacobian(f, x, fdjtype(backend))
end

function DI.value_and_jacobian(f, backend::AnyAutoFiniteDiff, x, ::NoJacobianExtras)
    y = f(x)
    return y, finite_difference_jacobian(f, x, fdjtype(backend), eltype(y), y)
end

function DI.jacobian!!(f, jac, backend::AnyAutoFiniteDiff, x, extras::NoJacobianExtras)
    return DI.jacobian(f, backend, x, extras)
end

function DI.value_and_jacobian!!(
    f, jac, backend::AnyAutoFiniteDiff, x, extras::NoJacobianExtras
)
    return DI.value_and_jacobian(f, backend, x, extras)
end

## Hessian

DI.prepare_hessian(f, ::AnyAutoFiniteDiff, x) = NoHessianExtras()

function DI.hessian(f, backend::AnyAutoFiniteDiff, x, ::NoHessianExtras)
    return finite_difference_hessian(f, x, fdhtype(backend))
end

function DI.hessian!!(f, hess, backend::AnyAutoFiniteDiff, x, ::NoHessianExtras)
    return finite_difference_hessian!(hess, f, x, fdhtype(backend))
end
