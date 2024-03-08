module DifferentiationInterfaceZygoteExt

using DifferentiationInterface: ChainRulesReverseBackend, ZygoteBackend
import DifferentiationInterface as DI
using DocStringExtensions
using Zygote: ZygoteRuleConfig, gradient, jacobian, withgradient, withjacobian

## Backend construction

"""
    ZygoteBackend(; custom=true)

Enables the use of [Zygote.jl](https://github.com/FluxML/Zygote.jl) by constructing a [`ChainRulesReverseBackend`](@ref) from `ZygoteRuleConfig()`.
"""
DI.ZygoteBackend(; custom::Bool=true) = ChainRulesReverseBackend(ZygoteRuleConfig(); custom)

const ZygoteBackendType{custom} = ChainRulesReverseBackend{custom,<:ZygoteRuleConfig}

function Base.show(io::IO, backend::ZygoteBackendType{custom}) where {custom}
    return print(io, "ZygoteBackend{$(custom ? "custom" : "fallback")}()")
end

## Utilities

function DI.value_and_gradient(::ZygoteBackendType{true}, f, x::AbstractArray)
    res = withgradient(f, x)
    return res.val, only(res.grad)
end

function DI.value_and_gradient!(
    grad::AbstractArray, backend::ZygoteBackendType{true}, f, x::AbstractArray
)
    y, new_grad = DI.value_and_gradient(backend, f, x)
    grad .= new_grad
    return y, grad
end

function DI.value_and_jacobian(::ZygoteBackendType{true}, f, x::AbstractArray)
    y = f(x)
    jac = jacobian(f, x)
    return y, only(jac)
end

function DI.value_and_jacobian!(
    jac::AbstractMatrix, backend::ZygoteBackendType{true}, f, x::AbstractArray
)
    y, new_jac = DI.value_and_jacobian(backend, f, x)
    jac .= new_jac
    return y, jac
end

end
