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

struct FiniteDiffOneArgDerivativeExtras{C}
    cache::C
end

function DI.prepare_derivative(f, backend::AnyAutoFiniteDiff, x)
    y = f(x)
    cache = if y isa Number
        nothing
    elseif y isa AbstractArray
        df = similar(y)
        cache = GradientCache(df, x, fdtype(backend), eltype(y), FUNCTION_NOT_INPLACE)
    end
    return FiniteDiffOneArgDerivativeExtras(cache)
end

### Scalar to scalar

function DI.derivative(
    f, backend::AnyAutoFiniteDiff, x, ::FiniteDiffOneArgDerivativeExtras{Nothing}
)
    return finite_difference_derivative(f, x, fdtype(backend))
end

function DI.derivative!(
    f,
    _der,
    backend::AnyAutoFiniteDiff,
    x,
    extras::FiniteDiffOneArgDerivativeExtras{Nothing},
)
    return DI.derivative(f, backend, x, extras)
end

function DI.value_and_derivative(
    f, backend::AnyAutoFiniteDiff, x, ::FiniteDiffOneArgDerivativeExtras{Nothing}
)
    y = f(x)
    return y, finite_difference_derivative(f, x, fdtype(backend), eltype(y), y)
end

function DI.value_and_derivative!(
    f,
    _der,
    backend::AnyAutoFiniteDiff,
    x,
    extras::FiniteDiffOneArgDerivativeExtras{Nothing},
)
    return DI.value_and_derivative(f, backend, x, extras)
end

### Scalar to array

function DI.derivative(
    f, ::AnyAutoFiniteDiff, x, extras::FiniteDiffOneArgDerivativeExtras{<:GradientCache}
)
    return finite_difference_gradient(f, x, extras.cache)
end

function DI.derivative!(
    f,
    der,
    ::AnyAutoFiniteDiff,
    x,
    extras::FiniteDiffOneArgDerivativeExtras{<:GradientCache},
)
    return finite_difference_gradient!(der, f, x, extras.cache)
end

function DI.value_and_derivative(
    f, ::AnyAutoFiniteDiff, x, extras::FiniteDiffOneArgDerivativeExtras{<:GradientCache}
)
    y = f(x)
    return y, finite_difference_gradient(f, x, extras.cache)
end

function DI.value_and_derivative!(
    f,
    der,
    ::AnyAutoFiniteDiff,
    x,
    extras::FiniteDiffOneArgDerivativeExtras{<:GradientCache},
)
    return f(x), finite_difference_gradient!(der, f, x, extras.cache)
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

function DI.gradient!(
    f, grad, ::AnyAutoFiniteDiff, x::AbstractArray, extras::FiniteDiffGradientExtras
)
    return finite_difference_gradient!(grad, f, x, extras.cache)
end

function DI.value_and_gradient!(
    f, grad, ::AnyAutoFiniteDiff, x::AbstractArray, extras::FiniteDiffGradientExtras
)
    return f(x), finite_difference_gradient!(grad, f, x, extras.cache)
end

## Jacobian

struct FiniteDiffOneArgJacobianExtras{C}
    cache::C
end

function DI.prepare_jacobian(f, backend::AnyAutoFiniteDiff, x)
    y = f(x)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    return FiniteDiffOneArgJacobianExtras(cache)
end

function DI.jacobian(f, ::AnyAutoFiniteDiff, x, extras::FiniteDiffOneArgJacobianExtras)
    return finite_difference_jacobian(f, x, extras.cache)
end

function DI.value_and_jacobian(
    f, ::AnyAutoFiniteDiff, x, extras::FiniteDiffOneArgJacobianExtras
)
    y = f(x)
    return y, finite_difference_jacobian(f, x, extras.cache, y)
end

function DI.jacobian!(
    f, jac, ::AnyAutoFiniteDiff, x, extras::FiniteDiffOneArgJacobianExtras
)
    return finite_difference_jacobian(f, x, extras.cache; jac_prototype=jac)
end

function DI.value_and_jacobian!(
    f, jac, ::AnyAutoFiniteDiff, x, extras::FiniteDiffOneArgJacobianExtras
)
    y = f(x)
    return y, finite_difference_jacobian(f, x, extras.cache, y; jac_prototype=jac)
end

## Hessian

struct FiniteDiffHessianExtras{C}
    cache::C
end

function DI.prepare_hessian(f, backend::AnyAutoFiniteDiff, x)
    cache = HessianCache(x, fdhtype(backend))
    return FiniteDiffHessianExtras(cache)
end

function DI.hessian(f, ::AnyAutoFiniteDiff, x, extras::FiniteDiffHessianExtras)
    return finite_difference_hessian(f, x, extras.cache)
end

function DI.hessian!(f, hess, ::AnyAutoFiniteDiff, x, extras::FiniteDiffHessianExtras)
    return finite_difference_hessian!(hess, f, x, extras.cache)
end
