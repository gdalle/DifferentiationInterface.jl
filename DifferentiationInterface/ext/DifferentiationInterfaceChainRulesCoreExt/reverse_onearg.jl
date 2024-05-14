## Pullback

struct ChainRulesPullbackExtrasSamePoint{Y,PB} <: PullbackExtras
    y::Y
    pb::PB
end

DI.prepare_pullback(f, ::AutoReverseChainRules, x, dy) = NoPullbackExtras()

function DI.prepare_pullback_same_point(
    f, backend::AutoReverseChainRules, x, dy, ::PullbackExtras=NoPullbackExtras()
)
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x)
    return ChainRulesPullbackExtrasSamePoint(y, pb)
end

function DI.value_and_pullback(f, backend::AutoReverseChainRules, x, dy, ::NoPullbackExtras)
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x)
    return y, last(pb(dy))
end

function DI.value_and_pullback(
    f, ::AutoReverseChainRules, x, dy, extras::ChainRulesPullbackExtrasSamePoint
)
    @compat (; y, pb) = extras
    return copy(y), last(pb(dy))
end

function DI.pullback(
    f, ::AutoReverseChainRules, x, dy, extras::ChainRulesPullbackExtrasSamePoint
)
    @compat (; pb) = extras
    return last(pb(dy))
end
