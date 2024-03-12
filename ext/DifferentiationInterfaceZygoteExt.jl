module DifferentiationInterfaceZygoteExt

using ADTypes: AutoZygote
using DifferentiationInterface: CustomImplem, update!
import DifferentiationInterface as DI
using DocStringExtensions
using Zygote: ZygoteRuleConfig, gradient, jacobian, pullback, withgradient, withjacobian

## Primitives

function DI.value_and_pullback!(dx, ::AutoZygote, f, x, dy, extras::Nothing=nothing)
    y, back = pullback(f, x)
    new_dx = only(back(dy))
    return y, update!(dx, new_dx)
end

function DI.value_and_pullback(::AutoZygote, f, x, dy, extras::Nothing=nothing)
    y, back = pullback(f, x)
    dx = only(back(dy))
    return y, dx
end

## Utilities

function DI.value_and_gradient(
    ::AutoZygote,
    f,
    x::AbstractArray,
    extras::Nothing=nothing,
    ::CustomImplem=CustomImplem(),
)
    res = withgradient(f, x)
    return res.val, only(res.grad)
end

function DI.value_and_gradient!(
    grad::AbstractArray,
    backend::AutoZygote,
    f,
    x::AbstractArray,
    extras=nothing,
    implem::CustomImplem=CustomImplem(),
)
    y, new_grad = DI.value_and_gradient(backend, f, x, extras, implem)
    grad .= new_grad
    return y, grad
end

function DI.value_and_jacobian(
    ::AutoZygote,
    f,
    x::AbstractArray,
    extras::Nothing=nothing,
    ::CustomImplem=CustomImplem(),
)
    y = f(x)
    jac = jacobian(f, x)
    return y, only(jac)
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix,
    backend::AutoZygote,
    f,
    x::AbstractArray,
    extras::Nothing=nothing,
    implem::CustomImplem=CustomImplem(),
)
    y, new_jac = DI.value_and_jacobian(backend, f, x, extras, implem)
    jac .= new_jac
    return y, jac
end

end
