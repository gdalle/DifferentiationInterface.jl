struct MooncakeOneArgPullbackPrep{Y,R} <: PullbackPrep
    y_prototype::Y
    rrule::R
end

function DI.prepare_pullback(
    f, backend::AutoMooncake, x, ty::DI.Tangents, contexts::Vararg{Context,C}
) where {C}
    y = f(x, map(unwrap, contexts)...)
    config = get_config(backend)
    rrule = build_rrule(
        get_interpreter(),
        Tuple{typeof(f),typeof(x),typeof.(map(unwrap, contexts))...};
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )
    prep = MooncakeOneArgPullbackPrep(y, rrule)
    DI.value_and_pullback(f, prep, backend, x, ty, contexts...)  # warm up
    return prep
end

function DI.value_and_pullback(
    f,
    prep::MooncakeOneArgPullbackPrep{Y},
    ::AutoMooncake,
    x,
    ty::DI.Tangents{1},
    contexts::Vararg{Context,C},
) where {Y,C}
    dy = only(ty)
    dy_righttype = convert(tangent_type(Y), dy)
    new_y, (_, new_dx) = value_and_pullback!!(
        prep.rrule, dy_righttype, f, x, map(unwrap, contexts)...
    )
    return new_y, DI.Tangents(new_dx)
end

function DI.value_and_pullback!(
    f,
    prep::MooncakeOneArgPullbackPrep{Y},
    tx::DI.Tangents,
    ::AutoMooncake,
    x,
    ty::DI.Tangents{1},
    contexts::Vararg{Context,C},
) where {Y,C}
    dx, dy = only(tx), only(ty)
    dy_righttype = convert(tangent_type(Y), dy)
    dx_righttype = set_to_zero!!(convert(tangent_type(typeof(x)), dx))
    contexts_coduals = map(zero_fcodual ∘ unwrap, contexts)
    y, (_, new_dx) = __value_and_pullback!!(
        prep.rrule,
        dy_righttype,
        zero_codual(f),
        CoDual(x, dx_righttype),
        contexts_coduals...,
    )
    copyto!(dx, new_dx)
    return y, tx
end

function DI.value_and_pullback(
    f,
    prep::MooncakeOneArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::DI.Tangents,
    contexts::Vararg{Context,C},
) where {C}
    ys_and_dxs = map(ty.d) do dy
        y, tx = DI.value_and_pullback(f, prep, backend, x, DI.Tangents(dy), contexts...)
        y, only(tx)
    end
    y = first(ys_and_dxs[1])
    dxs = last.(ys_and_dxs)
    return y, DI.Tangents(dxs...)
end

function DI.pullback(
    f,
    prep::MooncakeOneArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::DI.Tangents,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_pullback(f, prep, backend, x, ty, contexts...)[2]
end

function DI.pullback!(
    f,
    tx::DI.Tangents,
    prep::MooncakeOneArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::DI.Tangents,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_pullback!(f, tx, prep, backend, x, ty, contexts...)[2]
end
