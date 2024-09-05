struct TapirOneArgPullbackExtras{Y,R} <: PullbackExtras
    y_prototype::Y
    rrule::R
end

function DI.prepare_pullback(f, backend::AutoTapir, x, ty::Tangents)
    y = f(x)
    rrule = build_rrule(
        TapirInterpreter(),
        Tuple{typeof(f),typeof(x)};
        safety_on=backend.safe_mode,
        silence_safety_messages=false,
    )
    extras = TapirOneArgPullbackExtras(y, rrule)
    DI.value_and_pullback(f, extras, backend, x, ty)  # warm up
    return extras
end

function DI.value_and_pullback(
    f, extras::TapirOneArgPullbackExtras, backend::AutoTapir, x, ty::Tangents
)
    y = f(x)
    dxs = map(ty.d) do dy
        only(DI.pullback(f, extras, backend, x, SingleTangent(dy)))
    end
    return y, Tangents(dxs)
end

function DI.value_and_pullback(
    f, extras::TapirOneArgPullbackExtras{Y}, ::AutoTapir, x, ty::Tangents{1}
) where {Y}
    dy = only(ty)
    dy_righttype = convert(tangent_type(Y), dy)
    new_y, (_, new_dx) = value_and_pullback!!(extras.rrule, dy_righttype, f, x)
    return new_y, SingleTangent(new_dx)
end

function DI.value_and_pullback!(
    f, extras::TapirOneArgPullbackExtras{Y}, tx::Tangents, ::AutoTapir, x, ty::Tangents{1}
) where {Y}
    dx, dy = only(tx), only(ty)
    dy_righttype = convert(tangent_type(Y), dy)
    dx_righttype = set_to_zero!!(convert(tangent_type(typeof(x)), dx))
    y, (_, new_dx) = __value_and_pullback!!(
        extras.rrule, dy_righttype, zero_codual(f), CoDual(x, dx_righttype)
    )
    copyto!(dx, new_dx)
    return y, tx
end

function DI.pullback(
    f, extras::TapirOneArgPullbackExtras, backend::AutoTapir, x, ty::Tangents
)
    return DI.value_and_pullback(f, extras, backend, x, ty)[2]
end

function DI.pullback!(
    f, tx::Tangents, extras::TapirOneArgPullbackExtras, backend::AutoTapir, x, ty::Tangents
)
    return DI.value_and_pullback!(f, tx, extras, backend, x, ty)[2]
end
