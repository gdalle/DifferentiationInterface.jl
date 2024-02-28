module DifferentiationInterfaceChainRulesCoreExt

using ChainRulesCore
using DifferentiationInterface
using LinearAlgebra

ruleconfig(backend::ChainRulesForwardBackend) = backend.ruleconfig
ruleconfig(backend::ChainRulesReverseBackend) = backend.ruleconfig

function DifferentiationInterface.value_and_pushforward!(
    _dy::Y, backend::ChainRulesForwardBackend, f, x::X, dx::X
) where {X,Y<:Number}
    rc = ruleconfig(backend)
    y, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    return y, new_dy
end

function DifferentiationInterface.value_and_pushforward!(
    dy::Y, backend::ChainRulesForwardBackend, f, x::X, dx::X
) where {X,Y<:AbstractArray}
    rc = ruleconfig(backend)
    y, new_dy = frule_via_ad(rc, (NoTangent(), dx), f, x)
    dy .= new_dy
    return y, dy
end

function DifferentiationInterface.value_and_pullback!(
    _dx::X, backend::ChainRulesReverseBackend, f, x::X, dy::Y
) where {X<:Number,Y}
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    return y, new_dx
end

function DifferentiationInterface.value_and_pullback!(
    dx::X, backend::ChainRulesReverseBackend, f, x::X, dy::Y
) where {X<:AbstractArray,Y}
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    _, new_dx = pullback(dy)
    dx .= new_dx
    return y, dx
end

end
