module DifferentiationInterfaceChainRulesCoreExt

using ChainRulesCore: NoTangent, frule_via_ad, rrule_via_ad
using DifferentiationInterface: ChainRulesForwardBackend, ChainRulesReverseBackend
import DifferentiationInterface as DI
using DocStringExtensions

ruleconfig(backend::ChainRulesForwardBackend) = backend.ruleconfig
ruleconfig(backend::ChainRulesReverseBackend) = backend.ruleconfig

update!(_old::Number, new::Number) = new
update!(old, new) = old .= new

## Primitives

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward(backend::ChainRulesForwardBackend, f, x, dx)
    rc = ruleconfig(backend)
    y, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    return y, new_dy
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pushforward!(dy, backend::ChainRulesForwardBackend, f, x, dx)
    y, new_dy = DI.value_and_pushforward(backend, f, x, dx)
    return y, update!(dy, new_dy)
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pullback(backend::ChainRulesReverseBackend, f, x, dy)
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    return y, new_dx
end

"""
$(TYPEDSIGNATURES)
"""
function DI.value_and_pullback!(dx, backend::ChainRulesReverseBackend, f, x, dy)
    y, new_dx = DI.value_and_pullback(backend, f, x, dy)
    return y, update!(dx, new_dx)
end

end
