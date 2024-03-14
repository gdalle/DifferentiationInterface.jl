module DifferentiationInterfaceChainRulesCoreExt

using ADTypes: AutoChainRules
using ChainRulesCore:
    HasForwardsMode, HasReverseMode, NoTangent, RuleConfig, frule_via_ad, rrule_via_ad
using DifferentiationInterface: update!
import DifferentiationInterface as DI
using DocStringExtensions

ruleconfig(backend::AutoChainRules) = backend.ruleconfig

const AutoForwardChainRules = AutoChainRules{<:RuleConfig{>:HasForwardsMode}}
const AutoReverseChainRules = AutoChainRules{<:RuleConfig{>:HasReverseMode}}

DI.autodiff_mode(::AutoForwardChainRules) = DI.ForwardMode()
DI.autodiff_mode(::AutoReverseChainRules) = DI.ReverseMode()

## Primitives

function DI.value_and_pushforward(
    backend::AutoForwardChainRules, f, x, dx, extras::Nothing=nothing
)
    rc = ruleconfig(backend)
    y, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    return y, new_dy
end

function DI.value_and_pushforward!(
    dy::Union{Number,AbstractArray},
    backend::AutoForwardChainRules,
    f,
    x,
    dx,
    extras=nothing,
)
    y, new_dy = DI.value_and_pushforward(backend, f, x, dx, extras)
    return y, update!(dy, new_dy)
end

function DI.value_and_pullback(
    backend::AutoReverseChainRules, f, x, dy, extras::Nothing=nothing
)
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    return y, new_dx
end

function DI.value_and_pullback!(
    dx::Union{Number,AbstractArray},
    backend::AutoReverseChainRules,
    f,
    x,
    dy,
    extras=nothing,
)
    y, new_dx = DI.value_and_pullback(backend, f, x, dy, extras)
    return y, update!(dx, new_dx)
end

end
