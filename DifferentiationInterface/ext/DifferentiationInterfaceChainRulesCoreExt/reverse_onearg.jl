## Pullback

DI.prepare_pullback(f, ::AutoReverseChainRules, x, dy) = NoPullbackExtras()

function DI.value_and_pullback_split(
    f, backend::AutoReverseChainRules, x, ::NoPullbackExtras
)
    rc = ruleconfig(backend)
    y, pullback = rrule_via_ad(rc, f, x)
    pullbackfunc(dy) = last(pullback(dy))
    return y, pullbackfunc
end

function DI.value_and_pullback!_split(
    f, backend::AutoReverseChainRules, x, extras::NoPullbackExtras
)
    y, pullbackfunc = DI.value_and_pullback_split(f, backend, x, extras)
    pullbackfunc!(dx, dy) = copyto!(dx, pullbackfunc(dy))
    return y, pullbackfunc!
end

function DI.value_and_pullback(
    f, backend::AutoReverseChainRules, x, dy, extras::NoPullbackExtras
)
    y, pullbackfunc = DI.value_and_pullback_split(f, backend, x, extras)
    return y, pullbackfunc(dy)
end
