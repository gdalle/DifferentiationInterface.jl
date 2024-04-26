function ChainRulesCore.frule((_, dx), dw::DifferentiateWith, x)
    (; f, backend) = dw
    y, dy = DI.value_and_pushforward(f, backend, x, dx)
    return y, dy
end

function ChainRulesCore.rrule(dw::DifferentiateWith, x)
    (; f, backend) = dw
    y, pullbackfunc = DI.value_and_pullback_split(f, backend, x)
    pullbackfunc_adjusted(dy) = (NoTangent(), pullbackfunc(dy))
    return y, pullbackfunc_adjusted
end
