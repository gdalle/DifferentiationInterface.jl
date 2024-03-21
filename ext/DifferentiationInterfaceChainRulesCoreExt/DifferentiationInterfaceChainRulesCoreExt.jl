module DifferentiationInterfaceChainRulesCoreExt

using ADTypes: ADTypes, AutoChainRules
using ChainRulesCore:
    HasForwardsMode, HasReverseMode, NoTangent, RuleConfig, frule_via_ad, rrule_via_ad
using DifferentiationInterface: update!
import DifferentiationInterface as DI
using DocStringExtensions

ruleconfig(backend::AutoChainRules) = backend.ruleconfig

const AutoForwardChainRules = AutoChainRules{<:RuleConfig{>:HasForwardsMode}}
const AutoReverseChainRules = AutoChainRules{<:RuleConfig{>:HasReverseMode}}

DI.supports_mutation(::AutoChainRules) = DI.MutationNotSupported()
DI.mode(::AutoForwardChainRules) = ADTypes.AbstractForwardMode
DI.mode(::AutoReverseChainRules) = ADTypes.AbstractReverseMode

## Primitives

function DI.value_and_pushforward(f::F, backend::AutoForwardChainRules, x, dx) where {F}
    rc = ruleconfig(backend)
    y, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    return y, new_dy
end

function DI.value_and_pushforward!(
    f::F, dy, backend::AutoForwardChainRules, x, dx
) where {F}
    y, new_dy = DI.value_and_pushforward(f, backend, x, dx)
    return y, update!(dy, new_dy)
end

function DI.value_and_pullback(f::F, backend::AutoReverseChainRules, x, dy) where {F}
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    return y, new_dx
end

function DI.value_and_pullback!(f::F, dx, backend::AutoReverseChainRules, x, dy) where {F}
    y, new_dx = DI.value_and_pullback(f, backend, x, dy)
    return y, update!(dx, new_dx)
end

end
