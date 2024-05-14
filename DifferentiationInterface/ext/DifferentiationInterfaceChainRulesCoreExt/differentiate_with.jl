function ChainRulesCore.frule((_, dx), dw::DifferentiateWith, x)
    @compat (; f, backend) = dw
    y, dy = DI.value_and_pushforward(f, backend, x, dx)
    return y, dy
end

function ChainRulesCore.rrule(dw::DifferentiateWith, x)
    @compat (; f, backend) = dw
    y = f(x)
    extras_same = DI.prepare_pullback_same_point(f, backend, x, y)
    pullbackfunc(dy) = (NoTangent(), DI.pullback(f, backend, x, dy, extras_same))
    return y, pullbackfunc
end
