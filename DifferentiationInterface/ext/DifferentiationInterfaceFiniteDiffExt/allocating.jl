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

struct FiniteDiffAllocatingDerivativeExtras{C}
    cache::C
end

function DI.prepare_derivative(f, ::AnyAutoFiniteDiff, x)
    cache = nothing
    return FiniteDiffAllocatingDerivativeExtras(cache)
end

function DI.derivative(
    f, backend::AnyAutoFiniteDiff, x, ::FiniteDiffAllocatingDerivativeExtras
)
    return finite_difference_derivative(f, x, fdtype(backend))
end

function DI.value_and_derivative(
    f, backend::AnyAutoFiniteDiff, x, ::FiniteDiffAllocatingDerivativeExtras
)
    y = f(x)
    return y, finite_difference_derivative(f, x, fdtype(backend), eltype(y), y)
end

function DI.derivative!!(
    f, der, backend::AnyAutoFiniteDiff, x, extras::FiniteDiffAllocatingDerivativeExtras
)
    return DI.derivative(f, backend, x, extras)
end

function DI.value_and_derivative!!(
    f, der, backend::AnyAutoFiniteDiff, x, extras::FiniteDiffAllocatingDerivativeExtras
)
    return DI.value_and_derivative(f, backend, x, extras)
end

## Gradient

struct FiniteDiffGradientExtras{C}
    cache::C
end

function DI.prepare_gradient(f, backend::AnyAutoFiniteDiff, x)
    y = f(x)
    df = zero(y) .* x
    cache = GradientCache(df, x, fdtype(backend))
    return FiniteDiffGradientExtras(cache)
end

function DI.gradient(
    f, ::AnyAutoFiniteDiff, x::AbstractArray, extras::FiniteDiffGradientExtras
)
    return finite_difference_gradient(f, x, extras.cache)
end

function DI.value_and_gradient(
    f, ::AnyAutoFiniteDiff, x::AbstractArray, extras::FiniteDiffGradientExtras
)
    return f(x), finite_difference_gradient(f, x, extras.cache)
end

function DI.gradient!!(
    f, grad, ::AnyAutoFiniteDiff, x::AbstractArray, extras::FiniteDiffGradientExtras
)
    return finite_difference_gradient!(grad, f, x, extras.cache)
end

function DI.value_and_gradient!!(
    f, grad, ::AnyAutoFiniteDiff, x::AbstractArray, extras::FiniteDiffGradientExtras
)
    return f(x), finite_difference_gradient!(grad, f, x, extras.cache)
end

## Jacobian

struct FiniteDiffAllocatingJacobianExtras{C}
    cache::C
end

function DI.prepare_jacobian(f, backend::AnyAutoFiniteDiff, x)
    y = f(x)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    return FiniteDiffAllocatingJacobianExtras(cache)
end

function DI.jacobian(f, ::AnyAutoFiniteDiff, x, extras::FiniteDiffAllocatingJacobianExtras)
    return finite_difference_jacobian(f, x, extras.cache)
end

function DI.value_and_jacobian(
    f, ::AnyAutoFiniteDiff, x, extras::FiniteDiffAllocatingJacobianExtras
)
    y = f(x)
    return y, finite_difference_jacobian(f, x, extras.cache, y)
end

function DI.jacobian!!(
    f, jac, ::AnyAutoFiniteDiff, x, extras::FiniteDiffAllocatingJacobianExtras
)
    return finite_difference_jacobian(f, x, extras.cache; jac_prototype=jac)
end

function DI.value_and_jacobian!!(
    f, jac, ::AnyAutoFiniteDiff, x, extras::FiniteDiffAllocatingJacobianExtras
)
    y = f(x)
    return y, finite_difference_jacobian(f, x, extras.cache, y; jac_prototype=jac)
end

## Hessian

struct FiniteDiffHessianExtras{C}
    cache::C
end

function DI.prepare_hessian(f, backend::AnyAutoFiniteDiff, x)
    cache = HessianCache(x, fdtype(backend))
    return FiniteDiffHessianExtras(cache)
end

function DI.hessian(f, ::AnyAutoFiniteDiff, x, extras::FiniteDiffHessianExtras)
    return finite_difference_hessian(f, x, extras.cache)
end

function DI.hessian!!(f, hess, ::AnyAutoFiniteDiff, x, extras::FiniteDiffHessianExtras)
    return finite_difference_hessian!(hess, f, x, extras.cache)
end
