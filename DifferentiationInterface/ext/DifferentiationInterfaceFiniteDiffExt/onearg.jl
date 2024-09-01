## Pushforward

DI.prepare_pushforward(f, ::AutoFiniteDiff, x, tx::Tangents) = NoPushforwardExtras()

function DI.pushforward(f, ::NoPushforwardExtras, backend::AutoFiniteDiff, x, tx::Tangents)
    dys = map(tx.d) do dx
        step(t::Number) = f(x .+ t .* dx)
        finite_difference_derivative(step, zero(eltype(x)), fdtype(backend))
    end
    return Tangents(dys)
end

function DI.value_and_pushforward(
    f, ::NoPushforwardExtras, backend::AutoFiniteDiff, x, tx::Tangents
)
    y = f(x)
    dys = map(tx.d) do dx
        step(t::Number) = f(x .+ t .* dx)
        finite_difference_derivative(step, zero(eltype(x)), fdtype(backend), eltype(y), y)
    end
    return y, Tangents(dys)
end

## Derivative

struct FiniteDiffOneArgDerivativeExtras{C} <: DerivativeExtras
    cache::C
end

function DI.prepare_derivative(f, backend::AutoFiniteDiff, x)
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
    f, ::FiniteDiffOneArgDerivativeExtras{Nothing}, backend::AutoFiniteDiff, x
)
    return finite_difference_derivative(f, x, fdtype(backend))
end

function DI.value_and_derivative(
    f, ::FiniteDiffOneArgDerivativeExtras{Nothing}, backend::AutoFiniteDiff, x
)
    y = f(x)
    return y, finite_difference_derivative(f, x, fdtype(backend), eltype(y), y)
end

### Scalar to array

function DI.derivative(
    f, extras::FiniteDiffOneArgDerivativeExtras{<:GradientCache}, ::AutoFiniteDiff, x
)
    return finite_difference_gradient(f, x, extras.cache)
end

function DI.derivative!(
    f, der, extras::FiniteDiffOneArgDerivativeExtras{<:GradientCache}, ::AutoFiniteDiff, x
)
    return finite_difference_gradient!(der, f, x, extras.cache)
end

function DI.value_and_derivative(
    f, extras::FiniteDiffOneArgDerivativeExtras{<:GradientCache}, ::AutoFiniteDiff, x
)
    y = f(x)
    return y, finite_difference_gradient(f, x, extras.cache)
end

function DI.value_and_derivative!(
    f, der, extras::FiniteDiffOneArgDerivativeExtras{<:GradientCache}, ::AutoFiniteDiff, x
)
    return f(x), finite_difference_gradient!(der, f, x, extras.cache)
end

## Gradient

struct FiniteDiffGradientExtras{C} <: GradientExtras
    cache::C
end

function DI.prepare_gradient(f, backend::AutoFiniteDiff, x)
    y = f(x)
    df = zero(y) .* x
    cache = GradientCache(df, x, fdtype(backend))
    return FiniteDiffGradientExtras(cache)
end

function DI.gradient(
    f, extras::FiniteDiffGradientExtras, ::AutoFiniteDiff, x::AbstractArray
)
    return finite_difference_gradient(f, x, extras.cache)
end

function DI.value_and_gradient(
    f, extras::FiniteDiffGradientExtras, ::AutoFiniteDiff, x::AbstractArray
)
    return f(x), finite_difference_gradient(f, x, extras.cache)
end

function DI.gradient!(
    f, grad, extras::FiniteDiffGradientExtras, ::AutoFiniteDiff, x::AbstractArray
)
    return finite_difference_gradient!(grad, f, x, extras.cache)
end

function DI.value_and_gradient!(
    f, grad, extras::FiniteDiffGradientExtras, ::AutoFiniteDiff, x::AbstractArray
)
    return f(x), finite_difference_gradient!(grad, f, x, extras.cache)
end

## Jacobian

struct FiniteDiffOneArgJacobianExtras{C} <: JacobianExtras
    cache::C
end

function DI.prepare_jacobian(f, backend::AutoFiniteDiff, x)
    y = f(x)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    return FiniteDiffOneArgJacobianExtras(cache)
end

function DI.jacobian(f, extras::FiniteDiffOneArgJacobianExtras, ::AutoFiniteDiff, x)
    return finite_difference_jacobian(f, x, extras.cache)
end

function DI.value_and_jacobian(
    f, extras::FiniteDiffOneArgJacobianExtras, ::AutoFiniteDiff, x
)
    y = f(x)
    return y, finite_difference_jacobian(f, x, extras.cache, y)
end

function DI.jacobian!(f, jac, extras::FiniteDiffOneArgJacobianExtras, ::AutoFiniteDiff, x)
    return copyto!(jac, finite_difference_jacobian(f, x, extras.cache; jac_prototype=jac))
end

function DI.value_and_jacobian!(
    f, jac, extras::FiniteDiffOneArgJacobianExtras, ::AutoFiniteDiff, x
)
    y = f(x)
    return y,
    copyto!(jac, finite_difference_jacobian(f, x, extras.cache, y; jac_prototype=jac))
end

## Hessian

struct FiniteDiffHessianExtras{C1,C2} <: HessianExtras
    gradient_cache::C1
    hessian_cache::C2
end

function DI.prepare_hessian(f, backend::AutoFiniteDiff, x)
    y = f(x)
    df = zero(y) .* x
    gradient_cache = GradientCache(df, x, fdtype(backend))
    hessian_cache = HessianCache(x, fdhtype(backend))
    return FiniteDiffHessianExtras(gradient_cache, hessian_cache)
end

function DI.hessian(f, extras::FiniteDiffHessianExtras, backend::AutoFiniteDiff, x)
    return finite_difference_hessian(f, x, extras.hessian_cache)
end

function DI.hessian!(f, hess, extras::FiniteDiffHessianExtras, backend::AutoFiniteDiff, x)
    return finite_difference_hessian!(hess, f, x, extras.hessian_cache)
end

function DI.value_gradient_and_hessian(
    f, extras::FiniteDiffHessianExtras, backend::AutoFiniteDiff, x
)
    grad = finite_difference_gradient(f, x, extras.gradient_cache)
    hess = finite_difference_hessian(f, x, extras.hessian_cache)
    return f(x), grad, hess
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, extras::FiniteDiffHessianExtras, backend::AutoFiniteDiff, x
)
    finite_difference_gradient!(grad, f, x, extras.gradient_cache)
    finite_difference_hessian!(hess, f, x, extras.hessian_cache)
    return f(x), grad, hess
end
