module DifferentiationInterfaceChainRulesCoreExt

using ChainRulesCore
using DifferentiationInterface
using LinearAlgebra

ruleconfig(backend::ChainRulesBackend) = backend.ruleconfig

function DifferentiationInterface.pushforward!(
    dy::Y, backend::ChainRulesBackend{<:RuleConfig{>:HasForwardsMode}}, f, x::X, dx::X
) where {X<:AbstractArray,Y<:AbstractArray}
    rc = ruleconfig(backend)
    _, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    dy .= new_dy
    return dy
end

function DifferentiationInterface.pullback!(
    dx::X, backend::ChainRulesBackend{<:RuleConfig{>:HasReverseMode}}, f, x::X, dy::Y
) where {X<:AbstractArray,Y<:AbstractArray}
    rc = ruleconfig(backend)
    _, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    dx .= new_dx
    return dx
end

end
