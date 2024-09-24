module DifferentiationInterfaceFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
import DifferentiationInterface as DI
using DifferentiationInterface:
    NoGradientPrep, NoJacobianPrep, NoPullbackPrep, NoPushforwardPrep, Tangents
using FiniteDifferences: FiniteDifferences, grad, jacobian, jvp, j′vp
using LinearAlgebra: dot

DI.check_available(::AutoFiniteDifferences) = true
DI.inplace_support(::AutoFiniteDifferences) = DI.InPlaceNotSupported()

## Pushforward

function DI.prepare_pushforward(f, ::AutoFiniteDifferences, x, tx::Tangents)
    return NoPushforwardPrep()
end

function DI.pushforward(
    f, ::NoPushforwardPrep, backend::AutoFiniteDifferences, x, tx::Tangents
)
    ty = map(tx) do dx
        jvp(backend.fdm, f, (x, dx))
    end
    return ty
end

function DI.value_and_pushforward(
    f, prep::NoPushforwardPrep, backend::AutoFiniteDifferences, x, tx::Tangents
)
    return f(x), DI.pushforward(f, prep, backend, x, tx)
end

## Pullback

DI.prepare_pullback(f, ::AutoFiniteDifferences, x, ty::Tangents) = NoPullbackPrep()

function DI.pullback(f, ::NoPullbackPrep, backend::AutoFiniteDifferences, x, ty::Tangents)
    tx = map(ty) do dy
        only(j′vp(backend.fdm, f, dy, x))
    end
    return tx
end

function DI.value_and_pullback(
    f, prep::NoPullbackPrep, backend::AutoFiniteDifferences, x, ty::Tangents
)
    return f(x), DI.pullback(f, prep, backend, x, ty)
end

## Gradient

DI.prepare_gradient(f, ::AutoFiniteDifferences, x) = NoGradientPrep()

function DI.gradient(f, ::NoGradientPrep, backend::AutoFiniteDifferences, x)
    return only(grad(backend.fdm, f, x))
end

function DI.value_and_gradient(f, prep::NoGradientPrep, backend::AutoFiniteDifferences, x)
    return f(x), DI.gradient(f, prep, backend, x)
end

function DI.gradient!(f, grad, prep::NoGradientPrep, backend::AutoFiniteDifferences, x)
    return copyto!(grad, DI.gradient(f, prep, backend, x))
end

function DI.value_and_gradient!(
    f, grad, prep::NoGradientPrep, backend::AutoFiniteDifferences, x
)
    y, new_grad = DI.value_and_gradient(f, prep, backend, x)
    return y, copyto!(grad, new_grad)
end

## Jacobian

DI.prepare_jacobian(f, ::AutoFiniteDifferences, x) = NoJacobianPrep()

function DI.jacobian(f, ::NoJacobianPrep, backend::AutoFiniteDifferences, x)
    return only(jacobian(backend.fdm, f, x))
end

function DI.value_and_jacobian(f, prep::NoJacobianPrep, backend::AutoFiniteDifferences, x)
    return f(x), DI.jacobian(f, prep, backend, x)
end

function DI.jacobian!(f, jac, prep::NoJacobianPrep, backend::AutoFiniteDifferences, x)
    return copyto!(jac, DI.jacobian(f, prep, backend, x))
end

function DI.value_and_jacobian!(
    f, jac, prep::NoJacobianPrep, backend::AutoFiniteDifferences, x
)
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x)
    return y, copyto!(jac, new_jac)
end

end
