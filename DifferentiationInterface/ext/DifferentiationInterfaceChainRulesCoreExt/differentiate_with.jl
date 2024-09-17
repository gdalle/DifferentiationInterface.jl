function ChainRulesCore.rrule(dw::DifferentiateWith, x)
    @compat (; f, backend) = dw
    y = f(x)
    extras_same = DI.prepare_pullback_same_point(f, backend, x, DI.Tangents(y))
    function pullbackfunc(dy)
        tx = DI.pullback(f, extras_same, backend, x, DI.Tangents(dy))
        return (NoTangent(), only(tx))
    end
    return y, pullbackfunc
end
