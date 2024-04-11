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

function DI.value_and_pullback_split(f, ::AnyAutoZygote, x, ::NoPullbackExtras)
    y, back = pullback(f, x)
    pullbackfunc(dy) = only(back(dy))
    return y, pullbackfunc
end

function DI.value_and_pullback!_split(f, backend::AnyAutoZygote, x, ::NoPullbackExtras)
    y, pullbackfunc = DI.value_and_pullback_split(f, backend, x, extras)
    pullbackfunc!(dx, dy) = copyto!(dx, pullbackfunc(dy))
    return y, pullbackfunc!
end

function DI.value_and_pullback(f, backend::AnyAutoZygote, x, dy, extras::NoPullbackExtras)
    y, pullbackfunc = DI.value_and_pullback_split(f, backend, x, extras)
    return y, pullbackfunc(dy)
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

function DI.value_and_gradient!(
    f, grad, backend::AnyAutoZygote, x, extras::NoGradientExtras
)
    y, new_grad = DI.value_and_gradient(f, backend, x, extras)
    return y, copyto!(grad, new_grad)
end

function DI.gradient!(f, grad, backend::AnyAutoZygote, x, extras::NoGradientExtras)
    return copyto!(grad, DI.gradient(f, backend, x, extras))
end

## Jacobian

DI.prepare_jacobian(f, ::AnyAutoZygote, x) = NoJacobianExtras()

function DI.value_and_jacobian(f, ::AnyAutoZygote, x, ::NoJacobianExtras)
    return f(x), only(jacobian(f, x))  # https://github.com/FluxML/Zygote.jl/issues/1506
end

function DI.jacobian(f, ::AnyAutoZygote, x, ::NoJacobianExtras)
    return only(jacobian(f, x))
end

function DI.value_and_jacobian!(f, jac, backend::AnyAutoZygote, x, extras::NoJacobianExtras)
    y, new_jac = DI.value_and_jacobian(f, backend, x, extras)
    return y, copyto!(jac, new_jac)
end

function DI.jacobian!(f, jac, backend::AnyAutoZygote, x, extras::NoJacobianExtras)
    return copyto!(jac, DI.jacobian(f, backend, x, extras))
end

## Hessian

DI.prepare_hessian(f, ::AnyAutoZygote, x) = NoHessianExtras()

function DI.hessian(f, ::AnyAutoZygote, x, ::NoHessianExtras)
    return hessian(f, x)
end

function DI.hessian!(f, hess, backend::AnyAutoZygote, x, extras::NoHessianExtras)
    return copyto!(hess, DI.hessian(f, backend, x, extras))
end

end
