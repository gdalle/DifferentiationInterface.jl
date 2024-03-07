module DifferentiationInterfaceChainRulesCoreExt

using ChainRulesCore:
    HasForwardsMode, HasReverseMode, NoTangent, RuleConfig, frule_via_ad, rrule_via_ad
using DifferentiationInterface: ChainRulesForwardBackend, ChainRulesReverseBackend
import DifferentiationInterface as DI
using DocStringExtensions

ruleconfig(backend::ChainRulesForwardBackend) = backend.ruleconfig
ruleconfig(backend::ChainRulesReverseBackend) = backend.ruleconfig

update!(_old::Number, new::Number) = new
update!(old, new) = old .= new

## Backend construction

"""
$(SIGNATURES)
"""
function DI.ChainRulesForwardBackend(rc::RuleConfig{>:HasForwardsMode}; custom::Bool=true)
    return ChainRulesForwardBackend{custom,typeof(rc)}(rc)
end

"""
$(SIGNATURES)
"""
function DI.ChainRulesReverseBackend(rc::RuleConfig{>:HasReverseMode}; custom::Bool=true)
    return ChainRulesReverseBackend{custom,typeof(rc)}(rc)
end

## Primitives

function DI.value_and_pushforward(backend::ChainRulesForwardBackend, f, x, dx)
    rc = ruleconfig(backend)
    y, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    return y, new_dy
end

function DI.value_and_pushforward!(dy, backend::ChainRulesForwardBackend, f, x, dx)
    y, new_dy = DI.value_and_pushforward(backend, f, x, dx)
    return y, update!(dy, new_dy)
end

function DI.value_and_pullback(backend::ChainRulesReverseBackend, f, x, dy)
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    return y, new_dx
end

function DI.value_and_pullback!(dx, backend::ChainRulesReverseBackend, f, x, dy)
    y, new_dx = DI.value_and_pullback(backend, f, x, dy)
    return y, update!(dx, new_dx)
end

end
