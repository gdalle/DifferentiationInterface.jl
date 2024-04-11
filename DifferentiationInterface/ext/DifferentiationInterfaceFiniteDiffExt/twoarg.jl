## Pushforward

DI.prepare_pushforward(f!, ::AnyAutoFiniteDiff, y, x) = NoPushforwardExtras()

function DI.value_and_pushforward!(
    f!,
    (y, dy)::Tuple{<:AbstractArray,<:AbstractArray},
    backend::AnyAutoFiniteDiff,
    x,
    dx,
    ::NoPushforwardExtras,
)
    function step(t::Number)::AbstractArray
        new_y = similar(y)
        f!(new_y, x .+ t .* dx)
        return new_y
    end
    f!(y, x)
    new_dy = finite_difference_derivative(
        step, zero(eltype(x)), fdtype(backend), eltype(y), y
    )
    return y, copyto!(dy, new_dy)
end

## Derivative

struct FiniteDiffTwoArgDerivativeExtras{C}
    cache::C
end

function DI.prepare_derivative(f!, ::AnyAutoFiniteDiff, y, x)
    cache = nothing
    return FiniteDiffTwoArgDerivativeExtras(cache)
end

function DI.value_and_derivative!(
    f!,
    (y, der)::Tuple{<:AbstractArray,<:AbstractArray},
    backend::AnyAutoFiniteDiff,
    x,
    ::FiniteDiffTwoArgDerivativeExtras,
)
    f!(y, x)
    finite_difference_gradient!(der, f!, x, fdtype(backend), eltype(y), FUNCTION_INPLACE, y)
    return y, der
end

function DI.derivative!(
    f!,
    (y, der)::Tuple{<:AbstractArray,<:AbstractArray},
    backend::AnyAutoFiniteDiff,
    x,
    ::FiniteDiffTwoArgDerivativeExtras,
)
    finite_difference_gradient!(der, f!, x, fdtype(backend), eltype(y), FUNCTION_INPLACE)
    return der
end

## Jacobian

struct FiniteDiffTwoArgJacobianExtras{C}
    cache::C
end

function DI.prepare_jacobian(f!, backend::AnyAutoFiniteDiff, y, x)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    return FiniteDiffTwoArgJacobianExtras(cache)
end

function DI.value_and_jacobian!(
    f!,
    (y, jac)::Tuple{<:AbstractArray,<:AbstractMatrix},
    ::AnyAutoFiniteDiff,
    x,
    extras::FiniteDiffTwoArgJacobianExtras,
)
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    f!(y, x)
    return y, jac
end

function DI.jacobian!(
    f!,
    (y, jac)::Tuple{<:AbstractArray,<:AbstractMatrix},
    ::AnyAutoFiniteDiff,
    x,
    extras::FiniteDiffTwoArgJacobianExtras,
)
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    return jac
end
