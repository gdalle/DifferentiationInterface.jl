module DifferentiationInterfaceZygoteExt

using ADTypes: AutoForwardDiff, AutoZygote
import DifferentiationInterface as DI
using DifferentiationInterface:
    Context,
    HVPExtras,
    NoGradientExtras,
    NoHessianExtras,
    NoJacobianExtras,
    NoPullbackExtras,
    PullbackExtras,
    Tangents,
    unwrap
using ForwardDiff: ForwardDiff
using Zygote:
    ZygoteRuleConfig, gradient, hessian, jacobian, pullback, withgradient, withjacobian
using Compat

DI.check_available(::AutoZygote) = true
DI.inplace_support(::AutoZygote) = DI.InPlaceNotSupported()

## Pullback

struct ZygotePullbackExtrasSamePoint{Y,PB} <: PullbackExtras
    y::Y
    pb::PB
end

function DI.prepare_pullback(f, ::AutoZygote, x, ty::Tangents, contexts::Vararg{Context})
    return NoPullbackExtras()
end

function DI.prepare_pullback_same_point(
    f, ::NoPullbackExtras, ::AutoZygote, x, ty::Tangents, contexts::Vararg{Context}
)
    y, pb = pullback(f, x, map(unwrap, contexts)...)
    return ZygotePullbackExtrasSamePoint(y, pb)
end

function DI.value_and_pullback(
    f, ::NoPullbackExtras, ::AutoZygote, x, ty::Tangents, contexts::Vararg{Context}
)
    y, pb = pullback(f, x, map(unwrap, contexts)...)
    tx = map(ty) do dy
        first(pb(dy))
    end
    return y, tx
end

function DI.value_and_pullback(
    f,
    extras::ZygotePullbackExtrasSamePoint,
    ::AutoZygote,
    x,
    ty::Tangents,
    contexts::Vararg{Context},
)
    @compat (; y, pb) = extras
    tx = map(ty) do dy
        first(pb(dy))
    end
    return copy(y), tx
end

function DI.pullback(
    f,
    extras::ZygotePullbackExtrasSamePoint,
    ::AutoZygote,
    x,
    ty::Tangents,
    contexts::Vararg{Context},
)
    @compat (; pb) = extras
    tx = map(ty) do dy
        first(pb(dy))
    end
    return tx
end

## Gradient

DI.prepare_gradient(f, ::AutoZygote, x, contexts::Vararg{Context}) = NoGradientExtras()

function DI.value_and_gradient(
    f, ::NoGradientExtras, ::AutoZygote, x, contexts::Vararg{Context}
)
    @compat (; val, grad) = withgradient(f, x, map(unwrap, contexts)...)
    return val, first(grad)
end

function DI.gradient(f, ::NoGradientExtras, ::AutoZygote, x, contexts::Vararg{Context})
    return first(gradient(f, x, map(unwrap, contexts)...))
end

function DI.value_and_gradient!(
    f, grad, extras::NoGradientExtras, backend::AutoZygote, x, contexts::Vararg{Context}
)
    y, new_grad = DI.value_and_gradient(f, extras, backend, x, contexts...)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(
    f, grad, extras::NoGradientExtras, backend::AutoZygote, x, contexts::Vararg{Context}
)
    return copyto!(grad, DI.gradient(f, extras, backend, x, contexts...))
end

## Jacobian

DI.prepare_jacobian(f, ::AutoZygote, x) = NoJacobianExtras()

function DI.value_and_jacobian(f, ::NoJacobianExtras, ::AutoZygote, x)
    return f(x), only(jacobian(f, x))  # https://github.com/FluxML/Zygote.jl/issues/1506
end

function DI.jacobian(f, ::NoJacobianExtras, ::AutoZygote, x)
    return only(jacobian(f, x))
end

function DI.value_and_jacobian!(f, jac, extras::NoJacobianExtras, backend::AutoZygote, x)
    y, new_jac = DI.value_and_jacobian(f, extras, backend, x)
    return y, copyto!(jac, new_jac)
end

function DI.jacobian!(f, jac, extras::NoJacobianExtras, backend::AutoZygote, x)
    return copyto!(jac, DI.jacobian(f, extras, backend, x))
end

## HVP

# Beware, this uses ForwardDiff for the inner differentiation

struct ZygoteHVPExtras{G,PE} <: HVPExtras
    ∇f::G
    pushforward_extras::PE
end

function DI.prepare_hvp(f, ::AutoZygote, x, tx::Tangents)
    ∇f(x) = only(gradient(f, x))
    pushforward_extras = DI.prepare_pushforward(∇f, AutoForwardDiff(), x, tx)
    return ZygoteHVPExtras(∇f, pushforward_extras)
end

function DI.hvp(f, extras::ZygoteHVPExtras, ::AutoZygote, x, tx::Tangents)
    @compat (; ∇f, pushforward_extras) = extras
    return DI.pushforward(∇f, pushforward_extras, AutoForwardDiff(), x, tx)
end

function DI.hvp!(f, tg::Tangents, extras::ZygoteHVPExtras, ::AutoZygote, x, tx::Tangents)
    @compat (; ∇f, pushforward_extras) = extras
    return DI.pushforward!(∇f, tg, pushforward_extras, AutoForwardDiff(), x, tx)
end

## Hessian

DI.prepare_hessian(f, ::AutoZygote, x) = NoHessianExtras()

function DI.hessian(f, ::NoHessianExtras, ::AutoZygote, x)
    return hessian(f, x)
end

function DI.hessian!(f, hess, extras::NoHessianExtras, backend::AutoZygote, x)
    return copyto!(hess, DI.hessian(f, extras, backend, x))
end

function DI.value_gradient_and_hessian(f, extras::NoHessianExtras, backend::AutoZygote, x)
    y, grad = DI.value_and_gradient(f, NoGradientExtras(), backend, x)
    hess = DI.hessian(f, extras, backend, x)
    return y, grad, hess
end

function DI.value_gradient_and_hessian!(
    f, grad, hess, extras::NoHessianExtras, backend::AutoZygote, x
)
    y, _ = DI.value_and_gradient!(f, grad, NoGradientExtras(), backend, x)
    DI.hessian!(f, hess, extras, backend, x)
    return y, grad, hess
end

end
