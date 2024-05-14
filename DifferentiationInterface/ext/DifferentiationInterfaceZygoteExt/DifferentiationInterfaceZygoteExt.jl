module DifferentiationInterfaceZygoteExt

using ADTypes: AutoZygote
import DifferentiationInterface as DI
using DifferentiationInterface:
    NoGradientExtras, NoHessianExtras, NoJacobianExtras, NoPullbackExtras, PullbackExtras
using DocStringExtensions
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

DI.prepare_pullback(f, ::AutoZygote, x, dy) = NoPullbackExtras()

function DI.prepare_pullback_same_point(
    f, ::AutoZygote, x, dy, ::PullbackExtras=NoPullbackExtras()
)
    y, pb = pullback(f, x)
    return ZygotePullbackExtrasSamePoint(y, pb)
end

function DI.value_and_pullback(f, ::AutoZygote, x, dy, ::NoPullbackExtras)
    y, pb = pullback(f, x)
    return y, only(pb(dy))
end

function DI.value_and_pullback(
    f, ::AutoZygote, x, dy, extras::ZygotePullbackExtrasSamePoint
)
    @compat (; y, pb) = extras
    return copy(y), only(pb(dy))
end

function DI.pullback(f, ::AutoZygote, x, dy, extras::ZygotePullbackExtrasSamePoint)
    @compat (; pb) = extras
    return only(pb(dy))
end

## Gradient

DI.prepare_gradient(f, ::AutoZygote, x) = NoGradientExtras()

function DI.value_and_gradient(f, ::AutoZygote, x, ::NoGradientExtras)
    @compat (; val, grad) = withgradient(f, x)
    return val, only(grad)
end

function DI.gradient(f, ::AutoZygote, x, ::NoGradientExtras)
    return only(gradient(f, x))
end

function DI.value_and_gradient!(f, grad, backend::AutoZygote, x, extras::NoGradientExtras)
    y, new_grad = DI.value_and_gradient(f, backend, x, extras)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(f, grad, backend::AutoZygote, x, extras::NoGradientExtras)
    return copyto!(grad, DI.gradient(f, backend, x, extras))
end

## Jacobian

DI.prepare_jacobian(f, ::AutoZygote, x) = NoJacobianExtras()

function DI.value_and_jacobian(f, ::AutoZygote, x, ::NoJacobianExtras)
    return f(x), only(jacobian(f, x))  # https://github.com/FluxML/Zygote.jl/issues/1506
end

function DI.jacobian(f, ::AutoZygote, x, ::NoJacobianExtras)
    return only(jacobian(f, x))
end

function DI.value_and_jacobian!(f, jac, backend::AutoZygote, x, extras::NoJacobianExtras)
    y, new_jac = DI.value_and_jacobian(f, backend, x, extras)
    return y, copyto!(jac, new_jac)
end

function DI.jacobian!(f, jac, backend::AutoZygote, x, extras::NoJacobianExtras)
    return copyto!(jac, DI.jacobian(f, backend, x, extras))
end

## Hessian

DI.prepare_hessian(f, ::AutoZygote, x) = NoHessianExtras()

function DI.hessian(f, ::AutoZygote, x, ::NoHessianExtras)
    return hessian(f, x)
end

function DI.hessian!(f, hess, backend::AutoZygote, x, extras::NoHessianExtras)
    return copyto!(hess, DI.hessian(f, backend, x, extras))
end

end
