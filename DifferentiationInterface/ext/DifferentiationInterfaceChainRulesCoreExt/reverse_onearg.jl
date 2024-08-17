## Pullback

struct ChainRulesPullbackExtrasSamePoint{Y,PB} <: PullbackExtras
    y::Y
    pb::PB
end

DI.prepare_pullback(f, ::AutoReverseChainRules, x, ty::Tangents{1}) = NoPullbackExtras()

function DI.prepare_pullback_same_point(
    f,
    backend::AutoReverseChainRules,
    x,
    ty::Tangents{1},
    ::PullbackExtras=NoPullbackExtras(),
)
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x)
    return ChainRulesPullbackExtrasSamePoint(y, pb)
end

function DI.value_and_pullback(
    f, backend::AutoReverseChainRules, x, ty::Tangents{1}, ::NoPullbackExtras
)
    dy = only(ty)
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x)
    return y, Tangents(last(pb(dy)))
end

function DI.value_and_pullback(
    f,
    ::AutoReverseChainRules,
    x,
    ty::Tangents{1},
    extras::ChainRulesPullbackExtrasSamePoint,
)
    @compat (; y, pb) = extras
    dy = only(ty)
    return copy(y), Tangents(last(pb(dy)))
end

function DI.pullback(
    f,
    ::AutoReverseChainRules,
    x,
    ty::Tangents{1},
    extras::ChainRulesPullbackExtrasSamePoint,
)
    @compat (; pb) = extras
    dy = only(ty)
    return Tangents(last(pb(dy)))
end
