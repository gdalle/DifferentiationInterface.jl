function ChainRulesCore.rrule(dw::DifferentiateWith, x)
    @compat (; f, backend) = dw
    y = f(x)
    prep_same = DI.prepare_pullback_same_point(f, backend, x, DI.Tangents(y))
    function pullbackfunc(dy)
        tx = DI.pullback(f, prep_same, backend, x, DI.Tangents(dy))
        return (NoTangent(), only(tx))
    end
    return y, pullbackfunc
end
