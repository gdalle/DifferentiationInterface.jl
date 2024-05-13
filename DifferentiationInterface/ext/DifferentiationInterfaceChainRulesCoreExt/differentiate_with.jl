struct OneArgPullbackFunc{B,F,X,E}
    f::F
    backend::B
    x::X
    extras::E
end

function (pbf::OneArgPullbackFunc)(dy)
    @compat (; f, backend, x, extras) = pbf
    return pullback(f, backend, x, dy, extras)
end

function value_and_pullback_split(
    f::F,
    backend::AbstractADType,
    x,
    extras::PullbackExtras=prepare_pullback_same_point(f, backend, x, f(x)),
) where {F}
    return f(x), OneArgPullbackFunc(f, backend, x, extras)
end

function ChainRulesCore.frule((_, dx), dw::DifferentiateWith, x)
    @compat (; f, backend) = dw
    y, dy = DI.value_and_pushforward(f, backend, x, dx)
    return y, dy
end

function ChainRulesCore.rrule(dw::DifferentiateWith, x)
    @compat (; f, backend) = dw
    y, pullbackfunc = DI.value_and_pullback_split(f, backend, x)
    pullbackfunc_adjusted(dy) = (NoTangent(), pullbackfunc(dy))
    return y, pullbackfunc_adjusted
end
