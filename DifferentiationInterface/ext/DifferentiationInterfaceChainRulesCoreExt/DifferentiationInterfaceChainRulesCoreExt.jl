module DifferentiationInterfaceChainRulesCoreExt

using ADTypes: ADTypes, AutoChainRules
using ChainRulesCore:
    HasForwardsMode, HasReverseMode, NoTangent, RuleConfig, frule_via_ad, rrule_via_ad
import DifferentiationInterface as DI
using DifferentiationInterface: NoPullbackExtras, NoPushforwardExtras

ruleconfig(backend::AutoChainRules) = backend.ruleconfig

const AutoForwardChainRules = AutoChainRules{<:RuleConfig{>:HasForwardsMode}}
const AutoReverseChainRules = AutoChainRules{<:RuleConfig{>:HasReverseMode}}

DI.supports_mutation(::AutoChainRules) = DI.MutationNotSupported()
DI.mode(::AutoForwardChainRules) = ADTypes.AbstractForwardMode
DI.mode(::AutoReverseChainRules) = ADTypes.AbstractReverseMode

## Pushforward (unused)

#=
DI.prepare_pushforward(f, ::AutoForwardChainRules, x) = NoPushforwardExtras()

function DI.value_and_pushforward(
    f, backend::AutoForwardChainRules, x, dx, ::NoPushforwardExtras
)
    rc = ruleconfig(backend)
    y, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    return y, new_dy
end
=#

## Pullback

DI.prepare_pullback(f, ::AutoForwardChainRules, x) = NoPullbackExtras()

function DI.value_and_pullback(f, backend::AutoReverseChainRules, x, dy, ::NoPullbackExtras)
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    return y, new_dx
end

end
