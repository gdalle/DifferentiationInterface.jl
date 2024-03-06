module DifferentiationInterfaceChainRulesCoreExt

using ChainRulesCore: NoTangent, frule_via_ad, rrule_via_ad
using DifferentiationInterface
using DocStringExtensions

ruleconfig(backend::ChainRulesForwardBackend) = backend.ruleconfig
ruleconfig(backend::ChainRulesReverseBackend) = backend.ruleconfig

update!(_old::Number, new::Number) = new
update!(old, new) = old .= new

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pushforward!(
    dy::Y, backend::ChainRulesForwardBackend, f, x::X, dx
) where {X,Y}
    rc = ruleconfig(backend)
    y, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    return y, update!(dy, new_dy)
end

"""
$(TYPEDSIGNATURES)
"""
function DifferentiationInterface.value_and_pullback!(
    dx, backend::ChainRulesReverseBackend, f, x::X, dy::Y
) where {X,Y}
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    return y, update!(dx, new_dx)
end

end
