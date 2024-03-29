## Pushforward

function DI.pushforward(f, backend::AllAutoFiniteDiff, x, dx, extras::Nothing)
    step(t::Number) = f(x .+ t .* dx)
    new_dy = finite_difference_derivative(step, zero(eltype(x)), fdtype(backend))
    return new_dy
end

function DI.value_and_pushforward(f, backend::AllAutoFiniteDiff, x, dx, extras::Nothing)
    y = f(x)
    step(t::Number) = f(x .+ t .* dx)
    new_dy = finite_difference_derivative(
        step, zero(eltype(x)), fdtype(backend), eltype(y), y
    )
    return y, new_dy
end

## Derivative

function DI.derivative(f, backend::AllAutoFiniteDiff, x, extras::Nothing)
    return finite_difference_derivative(f, x, fdtype(backend))
end

function DI.value_and_derivative(f, backend::AllAutoFiniteDiff, x, extras::Nothing)
    y = f(x)
    return y, finite_difference_derivative(f, x, fdtype(backend), eltype(y), y)
end

## Gradient

function DI.gradient(f, backend::AllAutoFiniteDiff, x::Number, extras::Nothing)
    return DI.derivative(f, backend, x, extras)
end

function DI.value_and_gradient(f, backend::AllAutoFiniteDiff, x::Number, extras::Nothing)
    return DI.value_and_derivative(f, backend, x, extras)
end

function DI.gradient(f, backend::AllAutoFiniteDiff, x::AbstractArray, extras::Nothing)
    return finite_difference_gradient(f, x, fdtype(backend))
end

function DI.value_and_gradient(
    f, backend::AllAutoFiniteDiff, x::AbstractArray, extras::Nothing
)
    y = f(x)
    return y, finite_difference_gradient(f, x, fdtype(backend), typeof(y), y)
end

function DI.gradient!!(
    f, grad, backend::AllAutoFiniteDiff, x::AbstractArray, extras::Nothing
)
    return finite_difference_gradient!(grad, f, x, fdtype(backend))
end

function DI.value_and_gradient!!(
    f, grad, backend::AllAutoFiniteDiff, x::AbstractArray, extras::Nothing
)
    y = f(x)
    return y, finite_difference_gradient!(grad, f, x, fdtype(backend), typeof(y), y)
end

## Jacobian

function DI.jacobian(f, backend::AllAutoFiniteDiff, x, extras::Nothing)
    return finite_difference_jacobian(f, x, fdjtype(backend))
end

function DI.value_and_jacobian(f, backend::AllAutoFiniteDiff, x, extras::Nothing)
    y = f(x)
    return y, finite_difference_jacobian(f, x, fdjtype(backend), eltype(y), y)
end

function DI.jacobian!!(f, jac, backend::AllAutoFiniteDiff, x, extras::Nothing)
    return DI.jacobian(f, backend, x, extras)
end

function DI.value_and_jacobian!!(f, jac, backend::AllAutoFiniteDiff, x, extras::Nothing)
    return DI.value_and_jacobian(f, backend, x, extras)
end

## Hessian

function DI.hessian(f, backend::AllAutoFiniteDiff, x, extras::Nothing)
    return finite_difference_hessian(f, x, fdhtype(backend))
end

function DI.hessian!!(f, hess, backend::AllAutoFiniteDiff, x, extras::Nothing)
    return finite_difference_hessian!(hess, f, x, fdhtype(backend))
end
