## Pullback

DI.prepare_pullback(f, ::AutoReverseChainRules, x, dy) = NoPullbackExtras()

function DI.value_and_pullback(
    f, backend::AutoReverseChainRules, x, dy, extras::NoPullbackExtras
)
    rc = ruleconfig(backend)
    y, pullbackfunc = rrule_via_ad(rc, f, x)
    return y, last(pullbackfunc(dy))
end
