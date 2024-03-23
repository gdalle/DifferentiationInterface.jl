module DifferentiationInterfaceChainRulesCoreExt

using ADTypes: ADTypes, AutoChainRules
using ChainRulesCore:
    HasForwardsMode, HasReverseMode, NoTangent, RuleConfig, frule_via_ad, rrule_via_ad
using DifferentiationInterface: myupdate!!
import DifferentiationInterface as DI

ruleconfig(backend::AutoChainRules) = backend.ruleconfig

const AutoForwardChainRules = AutoChainRules{<:RuleConfig{>:HasForwardsMode}}
const AutoReverseChainRules = AutoChainRules{<:RuleConfig{>:HasReverseMode}}

DI.supports_mutation(::AutoChainRules) = DI.MutationNotSupported()
DI.mode(::AutoForwardChainRules) = ADTypes.AbstractForwardMode
DI.mode(::AutoReverseChainRules) = ADTypes.AbstractReverseMode

## Primitives

function DI.value_and_pushforward(
    f::F, backend::AutoForwardChainRules, x, dx, extras::Nothing
) where {F}
    rc = ruleconfig(backend)
    y, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    return y, new_dy
end

function DI.value_and_pushforward!!(
    f::F, dy, backend::AutoForwardChainRules, x, dx, extras
) where {F}
    y, new_dy = DI.value_and_pushforward(f, backend, x, dx, extras)
    return y, myupdate!!(dy, new_dy)
end

function DI.value_and_pullback(
    f::F, backend::AutoReverseChainRules, x, dy, extras::Nothing
) where {F}
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    return y, new_dx
end

function DI.value_and_pullback!!(
    f::F, dx, backend::AutoReverseChainRules, x, dy, extras
) where {F}
    y, new_dx = DI.value_and_pullback(f, backend, x, dy, extras)
    return y, myupdate!!(dx, new_dx)
end

end
