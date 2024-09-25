struct MooncakeOneArgPullbackPrep{Y,R} <: DI.PullbackPrep
    y_prototype::Y
    rrule::R
end

function DI.prepare_pullback(f, backend::AutoMooncake, x, ty::DI.Tangents)
    y = f(x)
    rrule = build_rrule(
        get_interpreter(),
        Tuple{typeof(f),typeof(x)};
        debug_mode=false,  # TODO: modify
        silence_debug_messages=false,
    )
    prep = MooncakeOneArgPullbackPrep(y, rrule)
    DI.value_and_pullback(f, prep, backend, x, ty)  # warm up
    return prep
end

function DI.value_and_pullback(
    f, prep::MooncakeOneArgPullbackPrep, backend::AutoMooncake, x, ty::DI.Tangents
)
    y = f(x)
    tx = map(ty) do dy
        only(DI.pullback(f, prep, backend, x, DI.Tangents(dy)))
    end
    return y, tx
end

function DI.value_and_pullback(
    f, prep::MooncakeOneArgPullbackPrep{Y}, ::AutoMooncake, x, ty::DI.Tangents{1}
) where {Y}
    dy = only(ty)
    dy_righttype = convert(tangent_type(Y), dy)
    new_y, (_, new_dx) = value_and_pullback!!(prep.rrule, dy_righttype, f, x)
    return new_y, DI.Tangents(new_dx)
end

function DI.value_and_pullback!(
    f,
    prep::MooncakeOneArgPullbackPrep{Y},
    tx::DI.Tangents,
    ::AutoMooncake,
    x,
    ty::DI.Tangents{1},
) where {Y}
    dx, dy = only(tx), only(ty)
    dy_righttype = convert(tangent_type(Y), dy)
    dx_righttype = set_to_zero!!(convert(tangent_type(typeof(x)), dx))
    y, (_, new_dx) = __value_and_pullback!!(
        prep.rrule, dy_righttype, zero_codual(f), CoDual(x, dx_righttype)
    )
    copyto!(dx, new_dx)
    return y, tx
end

function DI.pullback(
    f, prep::MooncakeOneArgPullbackPrep, backend::AutoMooncake, x, ty::DI.Tangents
)
    return DI.value_and_pullback(f, prep, backend, x, ty)[2]
end

function DI.pullback!(
    f,
    tx::DI.Tangents,
    prep::MooncakeOneArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::DI.Tangents,
)
    return DI.value_and_pullback!(f, tx, prep, backend, x, ty)[2]
end
