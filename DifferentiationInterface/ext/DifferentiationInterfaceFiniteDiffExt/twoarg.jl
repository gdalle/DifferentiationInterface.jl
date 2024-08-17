## Pushforward

DI.prepare_pushforward(f!, y, ::AutoFiniteDiff, x, tx::Tangents{1}) = NoPushforwardExtras()

function DI.value_and_pushforward(
    f!, y, backend::AutoFiniteDiff, x, tx::Tangents{1}, ::NoPushforwardExtras
)
    dx = only(tx)
    function step(t::Number)::AbstractArray
        new_y = similar(y)
        f!(new_y, x .+ t .* dx)
        return new_y
    end
    new_dy = finite_difference_derivative(
        step, zero(eltype(x)), fdtype(backend), eltype(y), y
    )
    f!(y, x)
    return y, Tangents(new_dy)
end

## Derivative

struct FiniteDiffTwoArgDerivativeExtras{C} <: DerivativeExtras
    cache::C
end

function DI.prepare_derivative(f!, y, ::AutoFiniteDiff, x)
    cache = nothing
    return FiniteDiffTwoArgDerivativeExtras(cache)
end

function DI.value_and_derivative(
    f!, y, backend::AutoFiniteDiff, x, ::FiniteDiffTwoArgDerivativeExtras
)
    f!(y, x)
    der = finite_difference_gradient(f!, x, fdtype(backend), eltype(y), FUNCTION_INPLACE, y)
    return y, der
end

function DI.value_and_derivative!(
    f!, y, der, backend::AutoFiniteDiff, x, ::FiniteDiffTwoArgDerivativeExtras
)
    f!(y, x)
    finite_difference_gradient!(der, f!, x, fdtype(backend), eltype(y), FUNCTION_INPLACE, y)
    return y, der
end

function DI.derivative(
    f!, y, backend::AutoFiniteDiff, x, ::FiniteDiffTwoArgDerivativeExtras
)
    f!(y, x)
    der = finite_difference_gradient(f!, x, fdtype(backend), eltype(y), FUNCTION_INPLACE, y)
    return der
end

function DI.derivative!(
    f!, y, der, backend::AutoFiniteDiff, x, ::FiniteDiffTwoArgDerivativeExtras
)
    finite_difference_gradient!(der, f!, x, fdtype(backend), eltype(y), FUNCTION_INPLACE)
    return der
end

## Jacobian

struct FiniteDiffTwoArgJacobianExtras{C} <: JacobianExtras
    cache::C
end

function DI.prepare_jacobian(f!, y, backend::AutoFiniteDiff, x)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    return FiniteDiffTwoArgJacobianExtras(cache)
end

function DI.value_and_jacobian(
    f!, y, ::AutoFiniteDiff, x, extras::FiniteDiffTwoArgJacobianExtras
)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    f!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!, y, jac, ::AutoFiniteDiff, x, extras::FiniteDiffTwoArgJacobianExtras
)
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    f!(y, x)
    return y, jac
end

function DI.jacobian(f!, y, ::AutoFiniteDiff, x, extras::FiniteDiffTwoArgJacobianExtras)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    return jac
end

function DI.jacobian!(
    f!, y, jac, ::AutoFiniteDiff, x, extras::FiniteDiffTwoArgJacobianExtras
)
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    return jac
end
