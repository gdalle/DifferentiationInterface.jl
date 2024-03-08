module DifferentiationInterfaceZygoteExt

using ADTypes: AutoZygote
using DifferentiationInterface: AutoChainRules
import DifferentiationInterface as DI
using DocStringExtensions
using Zygote: ZygoteRuleConfig, gradient, jacobian, withgradient, withjacobian

## Primitives

const zygote_chainrules_backend = AutoChainRules(ZygoteRuleConfig())

function DI.value_and_pushforward!(dy, ::AutoZygote, f, x, dx)
    return DI.value_and_pushforward!(dy, zygote_chainrules_backend, f, x, dx)
end

function DI.value_and_pushforward(::AutoZygote, f, x, dx)
    return DI.value_and_pushforward(zygote_chainrules_backend, f, x, dx)
end

function DI.value_and_pullback!(dx, ::AutoZygote, f, x, dy)
    return DI.value_and_pullback!(dx, zygote_chainrules_backend, f, x, dy)
end

function DI.value_and_pullback(::AutoZygote, f, x, dy)
    return DI.value_and_pullback(zygote_chainrules_backend, f, x, dy)
end

## Utilities

function DI.value_and_gradient(::AutoZygote, f, x::AbstractArray)
    res = withgradient(f, x)
    return res.val, only(res.grad)
end

function DI.value_and_gradient!(
    grad::AbstractArray, backend::AutoZygote, f, x::AbstractArray
)
    y, new_grad = DI.value_and_gradient(backend, f, x)
    grad .= new_grad
    return y, grad
end

function DI.value_and_jacobian(::AutoZygote, f, x::AbstractArray)
    y = f(x)
    jac = jacobian(f, x)
    return y, only(jac)
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, backend::AutoZygote, f, x::AbstractArray
)
    y, new_jac = DI.value_and_jacobian(backend, f, x)
    jac .= new_jac
    return y, jac
end

end
