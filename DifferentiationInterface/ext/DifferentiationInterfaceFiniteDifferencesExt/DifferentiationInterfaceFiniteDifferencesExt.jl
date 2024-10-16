module DifferentiationInterfaceFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
import DifferentiationInterface as DI
using DifferentiationInterface:
    Context,
    NoGradientPrep,
    NoJacobianPrep,
    NoPullbackPrep,
    NoPushforwardPrep,
    unwrap,
    with_contexts
using FiniteDifferences: FiniteDifferences, grad, jacobian, jvp, j′vp
using LinearAlgebra: dot

DI.check_available(::AutoFiniteDifferences) = true
DI.inplace_support(::AutoFiniteDifferences) = DI.InPlaceNotSupported()

## Pushforward

function DI.prepare_pushforward(
    f, ::AutoFiniteDifferences, x, tx::NTuple, contexts::Vararg{Context,C}
) where {C}
    return NoPushforwardPrep()
end

function DI.pushforward(
    f,
    ::NoPushforwardPrep,
    backend::AutoFiniteDifferences,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    ty = map(tx) do dx
        return jvp(backend.fdm, fc, (x, dx))
    end
    return ty
end

function DI.value_and_pushforward(
    f,
    prep::NoPushforwardPrep,
    backend::AutoFiniteDifferences,
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return f(x, map(unwrap, contexts)...),
    DI.pushforward(f, prep, backend, x, tx, contexts...)
end

## Pullback

function DI.prepare_pullback(
    f, ::AutoFiniteDifferences, x, ty::NTuple, contexts::Vararg{Context,C}
) where {C}
    return NoPullbackPrep()
end

function DI.pullback(
    f,
    ::NoPullbackPrep,
    backend::AutoFiniteDifferences,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    fc = with_contexts(f, contexts...)
    tx = map(ty) do dy
        return only(j′vp(backend.fdm, fc, dy, x))
    end
    return tx
end

function DI.value_and_pullback(
    f,
    prep::NoPullbackPrep,
    backend::AutoFiniteDifferences,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return f(x, map(unwrap, contexts)...), DI.pullback(f, prep, backend, x, ty, contexts...)
end

## Gradient

function DI.prepare_gradient(
    f, ::AutoFiniteDifferences, x, contexts::Vararg{Context,C}
) where {C}
    return NoGradientPrep()
end

function DI.gradient(
    f, ::NoGradientPrep, backend::AutoFiniteDifferences, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return only(grad(backend.fdm, fc, x))
end

function DI.value_and_gradient(
    f, prep::NoGradientPrep, backend::AutoFiniteDifferences, x, contexts::Vararg{Context,C}
) where {C}
    return f(x, map(unwrap, contexts)...), DI.gradient(f, prep, backend, x, contexts...)
end

function DI.gradient!(
    f,
    grad,
    prep::NoGradientPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return copyto!(grad, DI.gradient(f, prep, backend, x, contexts...))
end

function DI.value_and_gradient!(
    f,
    grad,
    prep::NoGradientPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{Context,C},
) where {C}
    y, new_grad = DI.value_and_gradient(f, prep, backend, x, contexts...)
    return y, copyto!(grad, new_grad)
end

## Jacobian

function DI.prepare_jacobian(
    f, ::AutoFiniteDifferences, x, contexts::Vararg{Context,C}
) where {C}
    return NoJacobianPrep()
end

function DI.jacobian(
    f, ::NoJacobianPrep, backend::AutoFiniteDifferences, x, contexts::Vararg{Context,C}
) where {C}
    fc = with_contexts(f, contexts...)
    return only(jacobian(backend.fdm, fc, x))
end

function DI.value_and_jacobian(
    f, prep::NoJacobianPrep, backend::AutoFiniteDifferences, x, contexts::Vararg{Context,C}
) where {C}
    return f(x, map(unwrap, contexts)...), DI.jacobian(f, prep, backend, x, contexts...)
end

function DI.jacobian!(
    f,
    jac,
    prep::NoJacobianPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return copyto!(jac, DI.jacobian(f, prep, backend, x, contexts...))
end

function DI.value_and_jacobian!(
    f,
    jac,
    prep::NoJacobianPrep,
    backend::AutoFiniteDifferences,
    x,
    contexts::Vararg{Context,C},
) where {C}
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x, contexts...)
    return y, copyto!(jac, new_jac)
end

end
