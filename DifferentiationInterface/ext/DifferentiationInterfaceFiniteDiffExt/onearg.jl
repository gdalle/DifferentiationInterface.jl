## Pushforward

DI.prepare_pushforward(f, ::AutoFiniteDiff, x, tx::Tangents) = NoPushforwardPrep()

function DI.pushforward(f, ::NoPushforwardPrep, backend::AutoFiniteDiff, x, tx::Tangents)
    ty = map(tx) do dx
        step(t::Number) = f(x .+ t .* dx)
        finite_difference_derivative(step, zero(eltype(x)), fdtype(backend))
    end
    return ty
end

function DI.value_and_pushforward(
    f, ::NoPushforwardPrep, backend::AutoFiniteDiff, x, tx::Tangents
)
    y = f(x)
    ty = map(tx) do dx
        step(t::Number) = f(x .+ t .* dx)
        finite_difference_derivative(step, zero(eltype(x)), fdtype(backend), eltype(y), y)
    end
    return y, ty
end

## Derivative

struct FiniteDiffOneArgDerivativePrep{C} <: DerivativePrep
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
    return FiniteDiffOneArgDerivativePrep(cache)
end

### Scalar to scalar

function DI.derivative(
    f, ::FiniteDiffOneArgDerivativePrep{Nothing}, backend::AutoFiniteDiff, x
)
    return finite_difference_derivative(f, x, fdtype(backend))
end

function DI.value_and_derivative(
    f, ::FiniteDiffOneArgDerivativePrep{Nothing}, backend::AutoFiniteDiff, x
)
    y = f(x)
    return y, finite_difference_derivative(f, x, fdtype(backend), eltype(y), y)
end

### Scalar to array

function DI.derivative(
    f, prep::FiniteDiffOneArgDerivativePrep{<:GradientCache}, ::AutoFiniteDiff, x
)
    return finite_difference_gradient(f, x, prep.cache)
end

function DI.derivative!(
    f, der, prep::FiniteDiffOneArgDerivativePrep{<:GradientCache}, ::AutoFiniteDiff, x
)
    return finite_difference_gradient!(der, f, x, prep.cache)
end

function DI.value_and_derivative(
    f, prep::FiniteDiffOneArgDerivativePrep{<:GradientCache}, ::AutoFiniteDiff, x
)
    y = f(x)
    return y, finite_difference_gradient(f, x, prep.cache)
end

function DI.value_and_derivative!(
    f, der, prep::FiniteDiffOneArgDerivativePrep{<:GradientCache}, ::AutoFiniteDiff, x
)
    return f(x), finite_difference_gradient!(der, f, x, prep.cache)
end

## Gradient

struct FiniteDiffGradientPrep{C} <: GradientPrep
    cache::C
end

function DI.prepare_gradient(f, backend::AutoFiniteDiff, x)
    y = f(x)
    df = zero(y) .* x
    cache = GradientCache(df, x, fdtype(backend))
    return FiniteDiffGradientPrep(cache)
end

function DI.gradient(f, prep::FiniteDiffGradientPrep, ::AutoFiniteDiff, x::AbstractArray)
    return finite_difference_gradient(f, x, prep.cache)
end

function DI.value_and_gradient(
    f, prep::FiniteDiffGradientPrep, ::AutoFiniteDiff, x::AbstractArray
)
    return f(x), finite_difference_gradient(f, x, prep.cache)
end

function DI.gradient!(
    f, grad, prep::FiniteDiffGradientPrep, ::AutoFiniteDiff, x::AbstractArray
)
    return finite_difference_gradient!(grad, f, x, prep.cache)
end

function DI.value_and_gradient!(
    f, grad, prep::FiniteDiffGradientPrep, ::AutoFiniteDiff, x::AbstractArray
)
    return f(x), finite_difference_gradient!(grad, f, x, prep.cache)
end

## Jacobian

struct FiniteDiffOneArgJacobianPrep{C} <: JacobianPrep
    cache::C
end

function DI.prepare_jacobian(f, backend::AutoFiniteDiff, x)
    y = f(x)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    return FiniteDiffOneArgJacobianPrep(cache)
end

function DI.jacobian(f, prep::FiniteDiffOneArgJacobianPrep, ::AutoFiniteDiff, x)
    return finite_difference_jacobian(f, x, prep.cache)
end

function DI.value_and_jacobian(f, prep::FiniteDiffOneArgJacobianPrep, ::AutoFiniteDiff, x)
    y = f(x)
    return y, finite_difference_jacobian(f, x, prep.cache, y)
end

function DI.jacobian!(f, jac, prep::FiniteDiffOneArgJacobianPrep, ::AutoFiniteDiff, x)
    return copyto!(jac, finite_difference_jacobian(f, x, prep.cache; jac_prototype=jac))
end

function DI.value_and_jacobian!(
    f, jac, prep::FiniteDiffOneArgJacobianPrep, ::AutoFiniteDiff, x
)
    y = f(x)
    return y,
    copyto!(jac, finite_difference_jacobian(f, x, prep.cache, y; jac_prototype=jac))
end

## Hessian

struct FiniteDiffHessianPrep{C1,C2} <: HessianPrep
    gradient_cache::C1
    hessian_cache::C2
end

function DI.prepare_hessian(f, backend::AutoFiniteDiff, x)
    y = f(x)
    df = zero(y) .* x
    gradient_cache = GradientCache(df, x, fdtype(backend))
    hessian_cache = HessianCache(x, fdhtype(backend))
    return FiniteDiffHessianPrep(gradient_cache, hessian_cache)
end

function DI.hessian(f, prep::FiniteDiffHessianPrep, backend::AutoFiniteDiff, x)
    return finite_difference_hessian(f, x, prep.hessian_cache)
end

function DI.hessian!(f, hess, prep::FiniteDiffHessianPrep, backend::AutoFiniteDiff, x)
    return finite_difference_hessian!(hess, f, x, prep.hessian_cache)
end

function DI.value_gradient_and_hessian(
    f, prep::FiniteDiffHessianPrep, backend::AutoFiniteDiff, x
)
    grad = finite_difference_gradient(f, x, prep.gradient_cache)
    hess = finite_difference_hessian(f, x, prep.hessian_cache)
    return f(x), grad, hess
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, prep::FiniteDiffHessianPrep, backend::AutoFiniteDiff, x
)
    finite_difference_gradient!(grad, f, x, prep.gradient_cache)
    finite_difference_hessian!(hess, f, x, prep.hessian_cache)
    return f(x), grad, hess
end
