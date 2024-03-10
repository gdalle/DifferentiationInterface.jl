module DifferentiationInterfaceZygoteExt

using ADTypes: AutoZygote
using DifferentiationInterface: AutoChainRules, CustomImplem, update!
import DifferentiationInterface as DI
using DocStringExtensions
using Zygote: ZygoteRuleConfig, gradient, jacobian, pullback, withgradient, withjacobian

## Primitives

const zygote_chainrules_backend = AutoChainRules(ZygoteRuleConfig())

function DI.value_and_pullback!(dx, ::AutoZygote, f, x, dy)
    y, back = pullback(f, x)
    new_dx = only(back(dy))
    return y, update!(dx, new_dx)
end

function DI.value_and_pullback(::AutoZygote, f, x, dy)
    y, back = pullback(f, x)
    dx = only(back(dy))
    return y, dx
end

## Utilities

function DI.value_and_gradient(::CustomImplem, ::AutoZygote, f, x::AbstractArray)
    res = withgradient(f, x)
    return res.val, only(res.grad)
end

function DI.value_and_gradient!(
    ::CustomImplem, grad::AbstractArray, backend::AutoZygote, f, x::AbstractArray
)
    y, new_grad = DI.value_and_gradient(backend, f, x)
    grad .= new_grad
    return y, grad
end

function DI.value_and_jacobian(::CustomImplem, ::AutoZygote, f, x::AbstractArray)
    y = f(x)
    jac = jacobian(f, x)
    return y, only(jac)
end

function DI.value_and_jacobian!(
    ::CustomImplem, jac::AbstractMatrix, backend::AutoZygote, f, x::AbstractArray
)
    y, new_jac = DI.value_and_jacobian(backend, f, x)
    jac .= new_jac
    return y, jac
end

end
