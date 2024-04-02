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

DI.prepare_derivative(f!, ::AnyAutoFiniteDiff, y, x) = NoDerivativeExtras()

function DI.value_and_derivative!!(
    f!,
    y::AbstractArray,
    der::AbstractArray,
    backend::AnyAutoFiniteDiff,
    x,
    ::NoDerivativeExtras,
)
    f!(y, x)
    finite_difference_gradient!(der, f!, x, fdtype(backend), eltype(y), FUNCTION_INPLACE, y)
    return y, der
end

## Jacobian

DI.prepare_jacobian(f!, ::AnyAutoFiniteDiff, y, x) = NoJacobianExtras()

function DI.value_and_jacobian!!(
    f!,
    y::AbstractArray,
    jac::AbstractMatrix,
    backend::AnyAutoFiniteDiff,
    x,
    ::NoJacobianExtras,
)
    f!(y, x)
    finite_difference_jacobian!(jac, f!, x, fdjtype(backend), eltype(y), y)
    return y, jac
end
