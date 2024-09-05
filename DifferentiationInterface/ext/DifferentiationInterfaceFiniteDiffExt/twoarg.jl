## Pushforward

DI.prepare_pushforward(f!, y, ::AutoFiniteDiff, x, tx::Tangents) = NoPushforwardExtras()

function DI.value_and_pushforward(
    f!, y, ::NoPushforwardExtras, backend::AutoFiniteDiff, x, tx::Tangents
)
    dys = map(tx.d) do dx
        function step(t::Number)::AbstractArray
            new_y = similar(y)
            f!(new_y, x .+ t .* dx)
            return new_y
        end
        finite_difference_derivative(step, zero(eltype(x)), fdtype(backend), eltype(y), y)
    end
    f!(y, x)
    return y, Tangents(dys)
end

## Derivative

struct FiniteDiffTwoArgDerivativeExtras{C} <: DerivativeExtras
    cache::C
end

function DI.prepare_derivative(f!, y, backend::AutoFiniteDiff, x)
    df = similar(y)
    cache = GradientCache(df, x, fdtype(backend), eltype(y), FUNCTION_INPLACE)
    return FiniteDiffTwoArgDerivativeExtras(cache)
end

function DI.value_and_derivative(
    f!, y, extras::FiniteDiffTwoArgDerivativeExtras, backend::AutoFiniteDiff, x
)
    f!(y, x)
    der = finite_difference_gradient(f!, x, extras.cache)
    return y, der
end

function DI.value_and_derivative!(
    f!, y, der, extras::FiniteDiffTwoArgDerivativeExtras, backend::AutoFiniteDiff, x
)
    f!(y, x)
    finite_difference_gradient!(der, f!, x, extras.cache)
    return y, der
end

function DI.derivative(
    f!, y, extras::FiniteDiffTwoArgDerivativeExtras, backend::AutoFiniteDiff, x
)
    f!(y, x)
    der = finite_difference_gradient(f!, x, extras.cache)
    return der
end

function DI.derivative!(
    f!, y, der, extras::FiniteDiffTwoArgDerivativeExtras, backend::AutoFiniteDiff, x
)
    finite_difference_gradient!(der, f!, x, extras.cache)
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
    f!, y, extras::FiniteDiffTwoArgJacobianExtras, ::AutoFiniteDiff, x
)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    f!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!, y, jac, extras::FiniteDiffTwoArgJacobianExtras, ::AutoFiniteDiff, x
)
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    f!(y, x)
    return y, jac
end

function DI.jacobian(f!, y, extras::FiniteDiffTwoArgJacobianExtras, ::AutoFiniteDiff, x)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    return jac
end

function DI.jacobian!(
    f!, y, jac, extras::FiniteDiffTwoArgJacobianExtras, ::AutoFiniteDiff, x
)
    finite_difference_jacobian!(jac, f!, x, extras.cache)
    return jac
end
