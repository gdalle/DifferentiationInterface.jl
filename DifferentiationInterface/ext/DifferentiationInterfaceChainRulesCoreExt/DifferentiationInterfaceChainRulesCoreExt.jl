module DifferentiationInterfaceChainRulesCoreExt

using ADTypes: ADTypes, AutoChainRules
using ChainRulesCore:
    ChainRulesCore,
    HasForwardsMode,
    HasReverseMode,
    NoTangent,
    RuleConfig,
    frule_via_ad,
    rrule_via_ad
import DifferentiationInterface as DI
using DifferentiationInterface: DifferentiateWith, NoPullbackExtras, NoPushforwardExtras
using Compat: @compat

ruleconfig(backend::AutoChainRules) = backend.ruleconfig

const AutoForwardChainRules = AutoChainRules{<:RuleConfig{>:HasForwardsMode}}
const AutoReverseChainRules = AutoChainRules{<:RuleConfig{>:HasReverseMode}}

DI.check_available(::AutoChainRules) = true
DI.twoarg_support(::AutoChainRules) = DI.TwoArgNotSupported()

include("reverse_onearg.jl")
include("differentiate_with.jl")

end
