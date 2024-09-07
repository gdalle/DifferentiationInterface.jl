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
    tx = map(ty) do dy
        last(pb(dy))
    end
    return y, tx
end

function DI.value_and_pullback(
    f, extras::ChainRulesPullbackExtrasSamePoint, ::AutoReverseChainRules, x, ty::Tangents
)
    @compat (; y, pb) = extras
    tx = map(ty) do dy
        last(pb(dy))
    end
    return copy(y), tx
end

function DI.pullback(
    f, extras::ChainRulesPullbackExtrasSamePoint, ::AutoReverseChainRules, x, ty::Tangents
)
    @compat (; pb) = extras
    tx = map(ty) do dy
        last(pb(dy))
    end
    return tx
end
