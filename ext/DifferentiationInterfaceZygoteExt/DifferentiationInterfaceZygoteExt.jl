module DifferentiationInterfaceZygoteExt

using ADTypes: AutoZygote, AutoSparseZygote
import DifferentiationInterface as DI
using DocStringExtensions
using Zygote:
    ZygoteRuleConfig, gradient, hessian, jacobian, pullback, withgradient, withjacobian

const AllAutoZygote = Union{AutoZygote,AutoSparseZygote}

DI.supports_mutation(::AllAutoZygote) = DI.MutationNotSupported()

## Pullback

function DI.value_and_pullback(f, ::AllAutoZygote, x, dy, extras::Nothing)
    y, back = pullback(f, x)
    dx = only(back(dy))
    return y, dx
end

## Gradient

function DI.value_and_gradient(f, ::AllAutoZygote, x, extras::Nothing)
    (; val, grad) = withgradient(f, x)
    return val, only(grad)
end

function DI.gradient(f, ::AllAutoZygote, x, extras::Nothing)
    return only(gradient(f, x))
end

function DI.value_and_gradient!!(f, grad, backend::AllAutoZygote, x, extras::Nothing)
    return DI.value_and_gradient(f, backend, x, extras)
end

function DI.gradient!!(f, grad, backend::AllAutoZygote, x, extras::Nothing)
    return DI.gradient(f, backend, x, extras)
end

## Jacobian

function DI.value_and_jacobian(f, ::AllAutoZygote, x, extras::Nothing)
    return f(x), only(jacobian(f, x))  # https://github.com/FluxML/Zygote.jl/issues/1506
end

function DI.jacobian(f, ::AllAutoZygote, x, extras::Nothing)
    return only(jacobian(f, x))
end

function DI.value_and_jacobian!!(f, jac, backend::AllAutoZygote, x, extras::Nothing)
    return DI.value_and_jacobian(f, backend, x, extras)
end

function DI.jacobian!!(f, jac, backend::AllAutoZygote, x, extras::Nothing)
    return DI.jacobian(f, backend, x, extras)
end

## Hessian

function DI.hessian(f, ::AllAutoZygote, x, extras::Nothing)
    return hessian(f, x)
end

end
