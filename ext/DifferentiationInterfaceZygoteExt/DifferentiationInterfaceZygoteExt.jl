module DifferentiationInterfaceZygoteExt

using ADTypes: AutoZygote, AutoSparseZygote
import DifferentiationInterface as DI
using DifferentiationInterface:
    NoGradientExtras, NoHessianExtras, NoJacobianExtras, NoPullbackExtras
using DocStringExtensions
using Zygote:
    ZygoteRuleConfig, gradient, hessian, jacobian, pullback, withgradient, withjacobian

const AnyAutoZygote = Union{AutoZygote,AutoSparseZygote}

DI.supports_mutation(::AnyAutoZygote) = DI.MutationNotSupported()

## Pullback

DI.prepare_pullback(f, ::AnyAutoZygote, x) = NoPullbackExtras()

function DI.value_and_pullback(f, ::AnyAutoZygote, x, dy, ::NoPullbackExtras)
    y, back = pullback(f, x)
    dx = only(back(dy))
    return y, dx
end

## Gradient

DI.prepare_gradient(f, ::AnyAutoZygote, x) = NoGradientExtras()

function DI.value_and_gradient(f, ::AnyAutoZygote, x, ::NoGradientExtras)
    (; val, grad) = withgradient(f, x)
    return val, only(grad)
end

function DI.gradient(f, ::AnyAutoZygote, x, ::NoGradientExtras)
    return only(gradient(f, x))
end

function DI.value_and_gradient!!(
    f, grad, backend::AnyAutoZygote, x, extras::NoGradientExtras
)
    return DI.value_and_gradient(f, backend, x, extras)
end

function DI.gradient!!(f, grad, backend::AnyAutoZygote, x, extras::NoGradientExtras)
    return DI.gradient(f, backend, x, extras)
end

## Jacobian

DI.prepare_jacobian(f, ::AnyAutoZygote, x) = NoJacobianExtras()

function DI.value_and_jacobian(f, ::AnyAutoZygote, x, ::NoJacobianExtras)
    return f(x), only(jacobian(f, x))  # https://github.com/FluxML/Zygote.jl/issues/1506
end

function DI.jacobian(f, ::AnyAutoZygote, x, ::NoJacobianExtras)
    return only(jacobian(f, x))
end

function DI.value_and_jacobian!!(
    f, jac, backend::AnyAutoZygote, x, extras::NoJacobianExtras
)
    return DI.value_and_jacobian(f, backend, x, extras)
end

function DI.jacobian!!(f, jac, backend::AnyAutoZygote, x, extras::NoJacobianExtras)
    return DI.jacobian(f, backend, x, extras)
end

## Hessian

DI.prepare_hessian(f, ::AnyAutoZygote, x) = NoHessianExtras()

function DI.hessian(f, ::AnyAutoZygote, x, ::NoHessianExtras)
    return hessian(f, x)
end

function DI.hessian!!(f, hess, backend::AnyAutoZygote, x, extras::NoHessianExtras)
    return DI.hessian(f, backend, x, extras)
end

end
