## Pullback

struct ChainRulesPullbackPrepSamePoint{Y,PB} <: PullbackPrep
    y::Y
    pb::PB
end

function DI.prepare_pullback(
    f, ::AutoReverseChainRules, x, ty::NTuple, contexts::Vararg{Constant,C}
) where {C}
    return NoPullbackPrep()
end

function DI.prepare_pullback_same_point(
    f,
    ::NoPullbackPrep,
    backend::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{Constant,C},
) where {C}
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x, map(unwrap, contexts)...)
    return ChainRulesPullbackPrepSamePoint(y, pb)
end

function DI.value_and_pullback(
    f,
    ::NoPullbackPrep,
    backend::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{Constant,C},
) where {C}
    rc = ruleconfig(backend)
    y, pb = rrule_via_ad(rc, f, x, map(unwrap, contexts)...)
    tx = map(ty) do dy
        pb(dy)[2]
    end
    return y, tx
end

function DI.value_and_pullback(
    f,
    prep::ChainRulesPullbackPrepSamePoint,
    ::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{Constant,C},
) where {C}
    (; y, pb) = prep
    tx = map(ty) do dy
        pb(dy)[2]
    end
    return copy(y), tx
end

function DI.pullback(
    f,
    prep::ChainRulesPullbackPrepSamePoint,
    ::AutoReverseChainRules,
    x,
    ty::NTuple,
    contexts::Vararg{Constant,C},
) where {C}
    (; pb) = prep
    tx = map(ty) do dy
        pb(dy)[2]
    end
    return tx
end
