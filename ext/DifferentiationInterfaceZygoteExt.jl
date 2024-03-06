module DifferentiationInterfaceZygoteExt

using DifferentiationInterface: ChainRulesReverseBackend
import DifferentiationInterface as DI
using DocStringExtensions
using Zygote: ZygoteRuleConfig, gradient, jacobian, withgradient, withjacobian

const ZygoteBackend = ChainRulesReverseBackend{<:ZygoteRuleConfig}

## Special cases

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_gradient(::ZygoteBackend, f, x::AbstractArray)
    res = withgradient(f, x)
    return res.val, only(res.grad)
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_gradient!(
    grad::AbstractArray, backend::ZygoteBackend, f, x::AbstractArray
)
    y, new_grad = DI.value_and_gradient(backend, f, x)
    grad .= new_grad
    return y, grad
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_jacobian(::ZygoteBackend, f, x::AbstractArray)
    y = f(x)
    jac = jacobian(f, x)
    return y, only(jac)
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_jacobian!(
    jac::AbstractMatrix, backend::ZygoteBackend, f, x::AbstractArray
)
    y, new_jac = DI.value_and_jacobian(backend, f, x)
    jac .= new_jac
    return y, jac
end

end
