## Pushforward

DI.prepare_pushforward(f!, ::AnyAutoFiniteDiff, y, x) = NoPushforwardExtras()

function DI.value_and_pushforward!!(
    f!,
    y::AbstractArray,
    dy::AbstractArray,
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
    return y, new_dy
end

## Derivative

struct FiniteDiffMutatingDerivativeExtras{C}
    cache::C
end

function DI.prepare_derivative(f!, ::AnyAutoFiniteDiff, y, x)
    cache = nothing
    return FiniteDiffMutatingDerivativeExtras(cache)
end

function DI.value_and_derivative!!(
    f!,
    y::AbstractArray,
    der::AbstractArray,
    backend::AnyAutoFiniteDiff,
    x,
    ::FiniteDiffMutatingDerivativeExtras,
)
    f!(y, x)
    finite_difference_gradient!(der, f!, x, fdtype(backend), eltype(y), FUNCTION_INPLACE, y)
    return y, der
end

## Jacobian

struct FiniteDiffMutatingJacobianExtras{C}
    cache::C
end

function DI.prepare_jacobian(f!, backend::AnyAutoFiniteDiff, y, x)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    return FiniteDiffMutatingJacobianExtras(cache)
end

function DI.value_and_jacobian!!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    ::AnyAutoFiniteDiff,
    x,
    extras::FiniteDiffMutatingJacobianExtras,
)
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    f!(y, x)
    return y, jac
end
