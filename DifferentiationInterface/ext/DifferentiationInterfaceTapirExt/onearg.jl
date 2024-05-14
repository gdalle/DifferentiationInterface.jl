struct TapirOneArgPullbackExtras{Y,R} <: PullbackExtras
    y_prototype::Y
    rrule::R
end

function DI.prepare_pullback(f, backend::AutoTapir, x, dy)
    y = f(x)
    rrule = build_rrule(
        TapirInterpreter(), Tuple{typeof(f), typeof(x)};
        safety_on=backend.safe_mode, silence_safety_messages=false,
    )
    extras = TapirOneArgPullbackExtras(y, rrule)
    DI.value_and_pullback(f, backend, x, dy, extras)  # warm up
    return extras
end

function DI.value_and_pullback(
    f, ::AutoTapir, x, dy, extras::TapirOneArgPullbackExtras{Y}
) where {Y}
    dy_righttype = convert(tangent_type(Y), dy)
    new_y, (_, new_dx) = value_and_pullback!!(extras.rrule, dy_righttype, f, x)
    return new_y, new_dx
end

function DI.value_and_pullback!(
    f, dx, ::AutoTapir, x, dy, extras::TapirOneArgPullbackExtras{Y}
) where {Y}
    dy_righttype = convert(tangent_type(Y), dy)
    dx_righttype = set_to_zero!!(convert(tangent_type(typeof(x)), dx))
    y, (_, new_dx) = __value_and_pullback!!(
        extras.rrule, dy_righttype, zero_codual(f), CoDual(x, dx_righttype)
    )
    return y, copyto!(dx, new_dx)
end

function DI.pullback(f, backend::AutoTapir, x, dy, extras::TapirOneArgPullbackExtras)
    return DI.value_and_pullback(f, backend, x, dy, extras)[2]
end

function DI.pullback!(f, dx, backend::AutoTapir, x, dy, extras::TapirOneArgPullbackExtras)
    return DI.value_and_pullback!(f, dx, backend, x, dy, extras)[2]
end
