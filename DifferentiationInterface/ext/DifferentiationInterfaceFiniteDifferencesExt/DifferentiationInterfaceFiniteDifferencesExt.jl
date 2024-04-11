module DifferentiationInterfaceFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
import DifferentiationInterface as DI
using DifferentiationInterface:
    NoGradientExtras, NoJacobianExtras, NoPullbackExtras, NoPushforwardExtras
using FillArrays: OneElement
using FiniteDifferences: FiniteDifferences, grad, jacobian, jvp, j′vp
using LinearAlgebra: dot

DI.supports_mutation(::AutoFiniteDifferences) = DI.MutationNotSupported()

function FiniteDifferences.to_vec(a::OneElement)  # TODO: remove type piracy (https://github.com/JuliaDiff/FiniteDifferences.jl/issues/141)
    return FiniteDifferences.to_vec(collect(a))
end

## Pushforward

DI.prepare_pushforward(f, ::AutoFiniteDifferences, x) = NoPushforwardExtras()

function DI.pushforward(f, backend::AutoFiniteDifferences, x, dx, ::NoPushforwardExtras)
    return jvp(backend.fdm, f, (x, dx))
end

function DI.value_and_pushforward(
    f, backend::AutoFiniteDifferences, x, dx, extras::NoPushforwardExtras
)
    return f(x), DI.pushforward(f, backend, x, dx, extras)
end

## Pullback

DI.prepare_pullback(f, ::AutoFiniteDifferences, x) = NoPullbackExtras()

function DI.pullback(f, backend::AutoFiniteDifferences, x, dy, ::NoPullbackExtras)
    return only(j′vp(backend.fdm, f, dy, x))
end

function DI.value_and_pullback(
    f, backend::AutoFiniteDifferences, x, dy, extras::NoPullbackExtras
)
    return f(x), DI.pullback(f, backend, x, dy, extras)
end

## Gradient

DI.prepare_gradient(f, ::AutoFiniteDifferences, x) = NoGradientExtras()

function DI.gradient(f, backend::AutoFiniteDifferences, x, ::NoGradientExtras)
    return only(grad(backend.fdm, f, x))
end

function DI.value_and_gradient(
    f, backend::AutoFiniteDifferences, x, extras::NoGradientExtras
)
    return f(x), DI.gradient(f, backend, x, extras)
end

function DI.gradient!(f, grad, backend::AutoFiniteDifferences, x, extras::NoGradientExtras)
    return copyto!(grad, DI.gradient(f, backend, x, extras))
end

function DI.value_and_gradient!(
    f, grad, backend::AutoFiniteDifferences, x, extras::NoGradientExtras
)
    y, new_grad = DI.value_and_gradient(f, backend, x, extras)
    return y, copyto!(grad, new_grad)
end

## Jacobian

DI.prepare_jacobian(f, ::AutoFiniteDifferences, x) = NoJacobianExtras()

function DI.jacobian(f, backend::AutoFiniteDifferences, x, ::NoJacobianExtras)
    return only(jacobian(backend.fdm, f, x))
end

function DI.value_and_jacobian(
    f, backend::AutoFiniteDifferences, x, extras::NoJacobianExtras
)
    return f(x), DI.jacobian(f, backend, x, extras)
end

function DI.jacobian!(f, jac, backend::AutoFiniteDifferences, x, extras::NoJacobianExtras)
    return copyto!(jac, DI.jacobian(f, backend, x, extras))
end

function DI.value_and_jacobian!(
    f, jac, backend::AutoFiniteDifferences, x, extras::NoJacobianExtras
)
    y, new_jac = DI.value_and_jacobian(f, backend, x)
    return y, copyto!(jac, new_jac)
end

end
