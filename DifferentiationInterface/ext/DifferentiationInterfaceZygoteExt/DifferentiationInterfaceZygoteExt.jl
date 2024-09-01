module DifferentiationInterfaceZygoteExt

using ADTypes: AutoForwardDiff, AutoZygote
import DifferentiationInterface as DI
using DifferentiationInterface:
    HVPExtras,
    NoGradientExtras,
    NoHessianExtras,
    NoJacobianExtras,
    NoPullbackExtras,
    PullbackExtras,
    Tangents
using DocStringExtensions
using ForwardDiff: ForwardDiff
using Zygote:
    ZygoteRuleConfig, gradient, hessian, jacobian, pullback, withgradient, withjacobian
using Compat

DI.check_available(::AutoZygote) = true
DI.twoarg_support(::AutoZygote) = DI.TwoArgNotSupported()

## Pullback

struct ZygotePullbackExtrasSamePoint{Y,PB} <: PullbackExtras
    y::Y
    pb::PB
end

DI.prepare_pullback(f, ::AutoZygote, x, ty::Tangents) = NoPullbackExtras()

function DI.prepare_pullback_same_point(
    f, ::NoPullbackExtras, ::AutoZygote, x, ty::Tangents
)
    y, pb = pullback(f, x)
    return ZygotePullbackExtrasSamePoint(y, pb)
end

function DI.value_and_pullback(f, ::NoPullbackExtras, ::AutoZygote, x, ty::Tangents)
    y, pb = pullback(f, x)
    dxs = map(ty.d) do dy
        only(pb(dy))
    end
    return y, Tangents(dxs)
end

function DI.value_and_pullback(
    f, extras::ZygotePullbackExtrasSamePoint, ::AutoZygote, x, ty::Tangents
)
    @compat (; y, pb) = extras
    dxs = map(ty.d) do dy
        only(pb(dy))
    end
    return copy(y), Tangents(dxs)
end

function DI.pullback(
    f, extras::ZygotePullbackExtrasSamePoint, ::AutoZygote, x, ty::Tangents
)
    @compat (; pb) = extras
    dxs = map(ty.d) do dy
        only(pb(dy))
    end
    return Tangents(dxs)
end

## Gradient

DI.prepare_gradient(f, ::AutoZygote, x) = NoGradientExtras()

function DI.value_and_gradient(f, ::NoGradientExtras, ::AutoZygote, x)
    @compat (; val, grad) = withgradient(f, x)
    return val, only(grad)
end

function DI.gradient(f, ::NoGradientExtras, ::AutoZygote, x)
    return only(gradient(f, x))
end

function DI.value_and_gradient!(f, grad, extras::NoGradientExtras, backend::AutoZygote, x)
    y, new_grad = DI.value_and_gradient(f, extras, backend, x)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(f, grad, extras::NoGradientExtras, backend::AutoZygote, x)
    return copyto!(grad, DI.gradient(f, extras, backend, x))
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

function DI.hvp!(f, extras::ZygoteHVPExtras, tg::Tangents, ::AutoZygote, x, tx::Tangents)
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
    DI.hessian!(f, hess, extras, x, backend)
    return y, grad, hess
end

end
