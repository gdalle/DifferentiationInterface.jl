module DifferentiationInterfaceFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
import DifferentiationInterface as DI
using DifferentiationInterface:
    NoGradientExtras, NoJacobianExtras, NoPullbackExtras, NoPushforwardExtras, Tangents
using FillArrays: OneElement
using FiniteDifferences: FiniteDifferences, grad, jacobian, jvp, j′vp
using LinearAlgebra: dot

DI.check_available(::AutoFiniteDifferences) = true
DI.twoarg_support(::AutoFiniteDifferences) = DI.TwoArgNotSupported()

## Pushforward

function DI.prepare_pushforward(f, ::AutoFiniteDifferences, x, tx::Tangents)
    return NoPushforwardExtras()
end

function DI.pushforward(
    f, ::NoPushforwardExtras, backend::AutoFiniteDifferences, x, tx::Tangents
)
    dys = map(tx.d) do dx
        jvp(backend.fdm, f, (x, dx))
    end
    return Tangents(dys)
end

function DI.value_and_pushforward(
    f, extras::NoPushforwardExtras, backend::AutoFiniteDifferences, x, tx::Tangents
)
    return f(x), DI.pushforward(f, extras, backend, x, tx)
end

## Pullback

DI.prepare_pullback(f, ::AutoFiniteDifferences, x, ty::Tangents) = NoPullbackExtras()

function DI.pullback(f, ::NoPullbackExtras, backend::AutoFiniteDifferences, x, ty::Tangents)
    dxs = map(ty.d) do dy
        only(j′vp(backend.fdm, f, dy, x))
    end
    return Tangents(dxs)
end

function DI.value_and_pullback(
    f, extras::NoPullbackExtras, backend::AutoFiniteDifferences, x, ty::Tangents
)
    return f(x), DI.pullback(f, extras, backend, x, ty)
end

## Gradient

DI.prepare_gradient(f, ::AutoFiniteDifferences, x) = NoGradientExtras()

function DI.gradient(f, ::NoGradientExtras, backend::AutoFiniteDifferences, x)
    return only(grad(backend.fdm, f, x))
end

function DI.value_and_gradient(
    f, extras::NoGradientExtras, backend::AutoFiniteDifferences, x
)
    return f(x), DI.gradient(f, extras, backend, x)
end

function DI.gradient!(f, grad, extras::NoGradientExtras, backend::AutoFiniteDifferences, x)
    return copyto!(grad, DI.gradient(f, extras, backend, x))
end

function DI.value_and_gradient!(
    f, grad, extras::NoGradientExtras, backend::AutoFiniteDifferences, x
)
    y, new_grad = DI.value_and_gradient(f, extras, backend, x)
    return y, copyto!(grad, new_grad)
end

## Jacobian

DI.prepare_jacobian(f, ::AutoFiniteDifferences, x) = NoJacobianExtras()

function DI.jacobian(f, ::NoJacobianExtras, backend::AutoFiniteDifferences, x)
    return only(jacobian(backend.fdm, f, x))
end

function DI.value_and_jacobian(
    f, extras::NoJacobianExtras, backend::AutoFiniteDifferences, x
)
    return f(x), DI.jacobian(f, extras, backend, x)
end

function DI.jacobian!(f, jac, extras::NoJacobianExtras, backend::AutoFiniteDifferences, x)
    return copyto!(jac, DI.jacobian(f, extras, backend, x))
end

function DI.value_and_jacobian!(
    f, jac, extras::NoJacobianExtras, backend::AutoFiniteDifferences, x
)
    y, new_jac = DI.value_and_jacobian(f, extras, backend, x)
    return y, copyto!(jac, new_jac)
end

end
