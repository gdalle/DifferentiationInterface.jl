module DifferentiationInterfaceChainRulesCoreExt

using ChainRulesCore:
    HasForwardsMode, HasReverseMode, NoTangent, RuleConfig, frule_via_ad, rrule_via_ad
using DifferentiationInterface: AutoChainRules, ruleconfig
import DifferentiationInterface as DI
using DocStringExtensions

update!(_old::Number, new::Number) = new
update!(old, new) = old .= new

const AutoForwardChainRules = AutoChainRules{<:RuleConfig{>:HasForwardsMode}}
const AutoReverseChainRules = AutoChainRules{<:RuleConfig{>:HasReverseMode}}

## Primitives

function DI.value_and_pushforward(backend::AutoForwardChainRules, f, x, dx)
    rc = ruleconfig(backend)
    y, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    return y, new_dy
end

function DI.value_and_pushforward!(dy, backend::AutoForwardChainRules, f, x, dx)
    y, new_dy = DI.value_and_pushforward(backend, f, x, dx)
    return y, update!(dy, new_dy)
end

function DI.value_and_pullback(backend::AutoReverseChainRules, f, x, dy)
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    return y, new_dx
end

function DI.value_and_pullback!(dx, backend::AutoReverseChainRules, f, x, dy)
    y, new_dx = DI.value_and_pullback(backend, f, x, dy)
    return y, update!(dx, new_dx)
end

end
