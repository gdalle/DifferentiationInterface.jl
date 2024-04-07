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

## Pullback

DI.prepare_pullback(f, ::AutoReverseChainRules, x) = NoPullbackExtras()

function DI.value_and_pullback_split(
    f, backend::AutoReverseChainRules, x, ::NoPullbackExtras
)
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    pullbackfunc(dy) = last(pullback(dy))
    return y, pullbackfunc
end

function DI.value_and_pullback(
    f, backend::AutoReverseChainRules, x, dy, extras::NoPullbackExtras
)
    y, pullbackfunc = DI.value_and_pullback_split(f, backend, x, extras)
    return y, pullbackfunc(dy)
end

end
