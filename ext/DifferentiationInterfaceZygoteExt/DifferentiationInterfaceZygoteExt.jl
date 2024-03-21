module DifferentiationInterfaceZygoteExt

using ADTypes: AutoZygote
using DifferentiationInterface: update!
import DifferentiationInterface as DI
using DocStringExtensions
using Zygote: ZygoteRuleConfig, gradient, jacobian, pullback, withgradient, withjacobian

DI.supports_mutation(::AutoZygote) = DI.MutationNotSupported()

## Primitives

function DI.value_and_pullback!(
    dx::Union{Number,AbstractArray}, ::AutoZygote, f, x, dy, extras::Nothing
)
    y, back = pullback(f, x)
    new_dx = only(back(dy))
    return y, update!(dx, new_dx)
end

function DI.value_and_pullback(::AutoZygote, f, x, dy, extras::Nothing)
    y, back = pullback(f, x)
    dx = only(back(dy))
    return y, dx
end

## Utilities

function DI.value_and_gradient(::AutoZygote, f, x::AbstractArray, extras::Nothing)
    res = withgradient(f, x)
    return res.val, only(res.grad)
end

function DI.value_and_gradient!(
    grad::AbstractArray, backend::AutoZygote, f, x::AbstractArray, extras=nothing
)
    y, new_grad = DI.value_and_gradient(backend, f, x, extras)
    grad .= new_grad
    return y, grad
end

function DI.value_and_jacobian(::AutoZygote, f, x::AbstractArray, extras::Nothing)
    y = f(x)
    jac = jacobian(f, x)
    return y, only(jac)
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, backend::AutoZygote, f, x::AbstractArray, extras::Nothing
)
    y, new_jac = DI.value_and_jacobian(backend, f, x, extras)
    jac .= new_jac
    return y, jac
end

end
