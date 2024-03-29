module DifferentiationInterfaceZygoteExt

using ADTypes: AutoZygote, AutoSparseZygote
import DifferentiationInterface as DI
using DocStringExtensions
using Zygote:
    ZygoteRuleConfig, gradient, hessian, jacobian, pullback, withgradient, withjacobian

const AnyAutoZygote = Union{AutoZygote,AutoSparseZygote}

DI.supports_mutation(::AnyAutoZygote) = DI.MutationNotSupported()

## Pullback

function DI.value_and_pullback(f, ::AnyAutoZygote, x, dy, extras::Nothing)
    y, back = pullback(f, x)
    dx = only(back(dy))
    return y, dx
end

## Gradient

function DI.value_and_gradient(f, ::AnyAutoZygote, x, extras::Nothing)
    (; val, grad) = withgradient(f, x)
    return val, only(grad)
end

function DI.gradient(f, ::AnyAutoZygote, x, extras::Nothing)
    return only(gradient(f, x))
end

function DI.value_and_gradient!!(f, grad, backend::AnyAutoZygote, x, extras::Nothing)
    return DI.value_and_gradient(f, backend, x, extras)
end

function DI.gradient!!(f, grad, backend::AnyAutoZygote, x, extras::Nothing)
    return DI.gradient(f, backend, x, extras)
end

## Jacobian

function DI.value_and_jacobian(f, ::AnyAutoZygote, x, extras::Nothing)
    return f(x), only(jacobian(f, x))  # https://github.com/FluxML/Zygote.jl/issues/1506
end

function DI.jacobian(f, ::AnyAutoZygote, x, extras::Nothing)
    return only(jacobian(f, x))
end

function DI.value_and_jacobian!!(f, jac, backend::AnyAutoZygote, x, extras::Nothing)
    return DI.value_and_jacobian(f, backend, x, extras)
end

function DI.jacobian!!(f, jac, backend::AnyAutoZygote, x, extras::Nothing)
    return DI.jacobian(f, backend, x, extras)
end

## Hessian

function DI.hessian(f, ::AnyAutoZygote, x, extras::Nothing)
    return hessian(f, x)
end

end
