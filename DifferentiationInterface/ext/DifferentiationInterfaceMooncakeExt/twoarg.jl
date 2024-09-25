struct MooncakeTwoArgPullbackPrep{R} <: PullbackPrep
    rrule::R
end

function DI.prepare_pullback(
    f!, y, backend::AutoMooncake, x, ty::DI.Tangents, contexts::Vararg{Context,C}
) where {C}
    config = get_config(backend)
    rrule = build_rrule(
        get_interpreter(),
        Tuple{typeof(f!),typeof(y),typeof(x),typeof.(map(unwrap, contexts))...};
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )
    prep = MooncakeTwoArgPullbackPrep(rrule)
    DI.value_and_pullback(f!, y, prep, backend, x, ty, contexts...)  # warm up
    return prep
end

function DI.value_and_pullback(
    f!,
    y,
    prep::MooncakeTwoArgPullbackPrep,
    ::AutoMooncake,
    x,
    ty::DI.Tangents{1},
    contexts::Vararg{Context,C},
) where {C}
    dy = only(ty)
    dy_righttype = convert(tangent_type(typeof(y)), copy(dy))
    dx_righttype = zero_tangent(x)

    # We want the VJP, not VJP + dx, so I'm going to zero-out `dx`. `set_to_zero!!` has the advantage
    # that it will also replace any immutable components of `dx` to zero.
    dx_righttype = set_to_zero!!(dx_righttype)

    # We want `dy` to correspond to the cotangent of `y` _after_
    # running the forwards-pass, so I'm going to take a copy, and zero-out the original.
    dy_righttype_backup = copy(dy_righttype)
    dy_righttype = set_to_zero!!(dy_righttype)
    contexts_coduals = map(zero_fcodual ∘ unwrap, contexts)

    # Mutate a copy of `y`, so that we can run the reverse-pass later on.
    y_copy = copy(y)

    # In case `f!` is a closure
    df! = zero_tangent(f!)

    # Run the forwards-pass.
    out, pb!! = prep.rrule(
        CoDual(f!, fdata(df!)),
        CoDual(y_copy, fdata(dy_righttype)),
        CoDual(x, fdata(dx_righttype)),
        contexts_coduals...,
    )

    # Verify that the output is non-differentiable.
    @assert primal(out) === nothing

    # Set the cotangent of `y` to be equal to the requested value.
    dy_righttype = increment!!(dy_righttype, dy_righttype_backup)

    # Record the state of `y` before running the reverse-pass.
    y = copyto!(y, y_copy)

    # Run the reverse-pass.
    _, _, new_dx = pb!!(NoRData())

    return y, DI.Tangents(tangent(fdata(dx_righttype), new_dx))
end

function DI.value_and_pullback(
    f!,
    y,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::DI.Tangents,
    contexts::Vararg{Context,C},
) where {C}
    tx = map(ty) do dy
        _, tx = DI.value_and_pullback(f!, y, prep, backend, x, DI.Tangents(dy), contexts...)
        only(tx)
    end
    return y, tx
end

function DI.pullback(
    f!,
    y,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::DI.Tangents,
    contexts::Vararg{Context,C},
) where {C}
    tx = map(ty) do dy
        _, tx = DI.value_and_pullback(f!, y, prep, backend, x, DI.Tangents(dy), contexts...)
        only(tx)
    end
    return tx
end

function DI.value_and_pullback!(
    f!,
    y,
    tx::DI.Tangents,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::DI.Tangents,
    contexts::Vararg{Context,C},
) where {C}
    y, new_tx = DI.value_and_pullback(f!, y, prep, backend, x, ty, contexts...)
    return y, copyto!(tx, new_tx)
end

function DI.pullback!(
    f!,
    y,
    tx::DI.Tangents,
    prep::MooncakeTwoArgPullbackPrep,
    backend::AutoMooncake,
    x,
    ty::DI.Tangents,
    contexts::Vararg{Context,C},
) where {C}
    new_tx = DI.pullback(f!, y, prep, backend, x, ty, contexts...)
    return copyto!(tx, new_tx)
end
