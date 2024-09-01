## Pullback

struct ChainRulesPullbackExtrasSamePoint{Y,PB} <: PullbackExtras
    y::Y
    pb::PB
end

DI.prepare_pullback(f, ::AutoReverseChainRules, x, ty::Tangents) = NoPullbackExtras()

function DI.prepare_pullback_same_point(
    f, ::NoPullbackExtras, backend::AutoReverseChainRules, x, ty::Tangents
)
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x)
    return ChainRulesPullbackExtrasSamePoint(y, pb)
end

function DI.value_and_pullback(
    f, ::NoPullbackExtras, backend::AutoReverseChainRules, x, ty::Tangents
)
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x)
    return y, Tangents(last.(pb.(ty.d)))
end

function DI.value_and_pullback(
    f, extras::ChainRulesPullbackExtrasSamePoint, ::AutoReverseChainRules, x, ty::Tangents
)
    @compat (; y, pb) = extras
    return copy(y), Tangents(last.(pb.(ty.d)))
end

function DI.pullback(
    f, extras::ChainRulesPullbackExtrasSamePoint, ::AutoReverseChainRules, x, ty::Tangents
)
    @compat (; pb) = extras
    return Tangents(last.(pb.(ty.d)))
end
