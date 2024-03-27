module DifferentiationInterfaceFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
import DifferentiationInterface as DI
using FillArrays: OneElement
using FiniteDifferences: FiniteDifferences, grad, jacobian, jvp, j′vp
using LinearAlgebra: dot

DI.supports_mutation(::AutoFiniteDifferences) = DI.MutationNotSupported()

function FiniteDifferences.to_vec(a::OneElement)  # TODO: remove type piracy (https://github.com/JuliaDiff/FiniteDifferences.jl/issues/141)
    return FiniteDifferences.to_vec(collect(a))
end

## Pushforward

function DI.pushforward(f, backend::AutoFiniteDifferences, x, dx, extras::Nothing)
    return jvp(backend.fdm, f, (x, dx))
end

function DI.value_and_pushforward(f, backend::AutoFiniteDifferences, x, dx, extras::Nothing)
    return f(x), DI.pushforward(f, backend, x, dx, extras)
end

## Pullback

function DI.pullback(f, backend::AutoFiniteDifferences, x, dy, extras::Nothing)
    return only(j′vp(backend.fdm, f, dy, x))
end

function DI.value_and_pullback(f, backend::AutoFiniteDifferences, x, dy, extras::Nothing)
    return f(x), DI.pullback(f, backend, x, dy, extras)
end

## Gradient

function DI.gradient(f, backend::AutoFiniteDifferences, x, extras::Nothing)
    return only(grad(backend.fdm, f, x))
end

function DI.value_and_gradient(f, backend::AutoFiniteDifferences, x, extras::Nothing)
    return f(x), DI.gradient(f, backend, x, extras)
end

function DI.gradient!!(f, grad, backend::AutoFiniteDifferences, x, extras::Nothing)
    return DI.gradient(f, backend, x, extras)
end

function DI.value_and_gradient!!(
    f, grad, backend::AutoFiniteDifferences, x, extras::Nothing
)
    return DI.value_and_gradient(f, backend, x)
end

## Jacobian

function DI.jacobian(f, backend::AutoFiniteDifferences, x, extras::Nothing)
    return only(jacobian(backend.fdm, f, x))
end

function DI.value_and_jacobian(f, backend::AutoFiniteDifferences, x, extras::Nothing)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.jacobian!!(f, jac, backend::AutoFiniteDifferences, x, extras::Nothing)
    return DI.jacobian(f, backend, x, extras)
end

function DI.value_and_jacobian!!(f, jac, backend::AutoFiniteDifferences, x, extras::Nothing)
    return DI.value_and_jacobian(f, backend, x)
end

end
