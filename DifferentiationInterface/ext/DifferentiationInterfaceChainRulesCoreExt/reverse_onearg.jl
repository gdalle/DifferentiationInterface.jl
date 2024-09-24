## Pullback

struct ChainRulesPullbackPrepSamePoint{Y,PB} <: PullbackPrep
    y::Y
    pb::PB
end

function DI.prepare_pullback(f, ::AutoReverseChainRules, x, ty::Tangents)
    return NoPullbackPrep()
end

function DI.prepare_pullback_same_point(
    f, ::NoPullbackPrep, backend::AutoReverseChainRules, x, ty::Tangents
)
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x)
    return ChainRulesPullbackPrepSamePoint(y, pb)
end

function DI.value_and_pullback(
    f, ::NoPullbackPrep, backend::AutoReverseChainRules, x, ty::Tangents
)
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x)
    tx = map(ty) do dy
        pb(dy)[2]
    end
    return y, tx
end

function DI.value_and_pullback(
    f, prep::ChainRulesPullbackPrepSamePoint, ::AutoReverseChainRules, x, ty::Tangents
)
    @compat (; y, pb) = prep
    tx = map(ty) do dy
        pb(dy)[2]
    end
    return copy(y), tx
end

function DI.pullback(
    f, prep::ChainRulesPullbackPrepSamePoint, ::AutoReverseChainRules, x, ty::Tangents
)
    @compat (; pb) = prep
    tx = map(ty) do dy
        pb(dy)[2]
    end
    return tx
end
