struct MooncakeTwoArgPullbackPrep{R,F,Y,DX,DY} <: PullbackPrep
    rrule::R
    df!::F
    y_copy::Y
    dx_righttype::DX
    dy_righttype::DY
    dy_righttype_after::DY
end

function DI.prepare_pullback(
    f!, y, backend::AutoMooncake, x, ty::NTuple, contexts::Vararg{Context,C}
) where {C}
    config = get_config(backend)
    rrule = build_rrule(
        get_interpreter(),
        Tuple{typeof(f!),typeof(y),typeof(x),typeof.(map(unwrap, contexts))...};
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )
    df! = zero_tangent(f!)
    y_copy = copy(y)
    dx_righttype = zero_tangent(x)
    dy_righttype = zero_tangent(y)
    dy_righttype_after = zero_tangent(y)
    prep = MooncakeTwoArgPullbackPrep(
        rrule, df!, y_copy, dx_righttype, dy_righttype, dy_righttype_after
    )
    DI.value_and_pullback(f!, y, prep, backend, x, ty, contexts...)  # warm up
    return prep
end

function DI.value_and_pullback(
    f!,
    y,
    prep::MooncakeTwoArgPullbackPrep,
    ::AutoMooncake,
    x,
    ty::NTuple{1},
    contexts::Vararg{Context,C},
) where {C}
    dy = only(ty)

    # Set all tangent storage to zero.
    df! = set_to_zero!!(prep.df!)
    # dx_righttype = set_to_zero!!(prep.dx_righttype)  # TODO: why doesn't this work?
    dx_righttype = zero_tangent(x)
    dy_righttype = set_to_zero!!(prep.dy_righttype)

    # Prepare cotangent to add after the forward pass.
    dy_righttype_after = copyto!(prep.dy_righttype_after, dy)

    contexts_coduals = map(zero_fcodual ∘ unwrap, contexts)

    # Run the forward pass
    out, pb!! = prep.rrule(
        CoDual(f!, fdata(df!)),
        CoDual(prep.y_copy, fdata(dy_righttype)),
        CoDual(x, fdata(dx_righttype)),
        contexts_coduals...,
    )

    # Verify that the output is non-differentiable.
    @assert primal(out) === nothing

    # Increment the desired cotangent dy.
    dy_righttype = increment!!(dy_righttype, dy_righttype_after)

    # Record the state of y before running the reverse pass.
    y = copyto!(y, prep.y_copy)

    # Run the reverse pass.
    _, _, new_dx = pb!!(NoRData())

    return y, (tangent(fdata(dx_righttype), new_dx),)
end

function DI.value_and_pullback(
    f!,
    y,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    tx = map(ty) do dy
        _, tx = DI.value_and_pullback(f!, y, prep, backend, x, (dy,), contexts...)
        only(tx)
    end
    return y, tx
end

function DI.value_and_pullback!(
    f!,
    y,
    tx::NTuple,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    _, new_tx = DI.value_and_pullback(f!, y, prep, backend, x, ty, contexts...)
    foreach(copyto!, tx, new_tx)
    return y, tx
end

function DI.pullback(
    f!,
    y,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_pullback(f!, y, prep, backend, x, ty, contexts...)[2]
end

function DI.pullback!(
    f!,
    y,
    tx::NTuple,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_pullback!(f!, y, tx, prep, backend, x, ty, contexts...)[2]
end
