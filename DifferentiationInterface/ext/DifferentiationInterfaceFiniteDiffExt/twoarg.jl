## Pushforward

DI.prepare_pushforward(f!, y, ::AutoFiniteDiff, x, tx::Tangents) = NoPushforwardPrep()

function DI.value_and_pushforward(
    f!, y, ::NoPushforwardPrep, backend::AutoFiniteDiff, x, tx::Tangents
)
    ty = map(tx) do dx
        function step(t::Number)::AbstractArray
            new_y = similar(y)
            f!(new_y, x .+ t .* dx)
            return new_y
        end
        finite_difference_derivative(step, zero(eltype(x)), fdtype(backend), eltype(y), y)
    end
    f!(y, x)
    return y, ty
end

## Derivative

struct FiniteDiffTwoArgDerivativePrep{C} <: DerivativePrep
    cache::C
end

function DI.prepare_derivative(f!, y, backend::AutoFiniteDiff, x)
    df = similar(y)
    cache = GradientCache(df, x, fdtype(backend), eltype(y), FUNCTION_INPLACE)
    return FiniteDiffTwoArgDerivativePrep(cache)
end

function DI.value_and_derivative(
    f!, y, prep::FiniteDiffTwoArgDerivativePrep, backend::AutoFiniteDiff, x
)
    f!(y, x)
    der = finite_difference_gradient(f!, x, prep.cache)
    return y, der
end

function DI.value_and_derivative!(
    f!, y, der, prep::FiniteDiffTwoArgDerivativePrep, backend::AutoFiniteDiff, x
)
    f!(y, x)
    finite_difference_gradient!(der, f!, x, prep.cache)
    return y, der
end

function DI.derivative(
    f!, y, prep::FiniteDiffTwoArgDerivativePrep, backend::AutoFiniteDiff, x
)
    f!(y, x)
    der = finite_difference_gradient(f!, x, prep.cache)
    return der
end

function DI.derivative!(
    f!, y, der, prep::FiniteDiffTwoArgDerivativePrep, backend::AutoFiniteDiff, x
)
    finite_difference_gradient!(der, f!, x, prep.cache)
    return der
end

## Jacobian

struct FiniteDiffTwoArgJacobianPrep{C} <: JacobianPrep
    cache::C
end

function DI.prepare_jacobian(f!, y, backend::AutoFiniteDiff, x)
    x1 = similar(x)
    fx = similar(y)
    fx1 = similar(y)
    cache = JacobianCache(x1, fx, fx1, fdjtype(backend))
    return FiniteDiffTwoArgJacobianPrep(cache)
end

function DI.value_and_jacobian(
    f!, y, prep::FiniteDiffTwoArgJacobianPrep, ::AutoFiniteDiff, x
)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, f!, x, prep.cache)
    f!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!, y, jac, prep::FiniteDiffTwoArgJacobianPrep, ::AutoFiniteDiff, x
)
    finite_difference_jacobian!(jac, f!, x, prep.cache)
    f!(y, x)
    return y, jac
end

function DI.jacobian(f!, y, prep::FiniteDiffTwoArgJacobianPrep, ::AutoFiniteDiff, x)
    jac = similar(y, length(y), length(x))
    finite_difference_jacobian!(jac, f!, x, prep.cache)
    return jac
end

function DI.jacobian!(f!, y, jac, prep::FiniteDiffTwoArgJacobianPrep, ::AutoFiniteDiff, x)
    finite_difference_jacobian!(jac, f!, x, prep.cache)
    return jac
end
