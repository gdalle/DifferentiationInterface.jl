struct TapirOneArgPullbackExtras{Y,R} <: PullbackExtras
    y_prototype::Y
    rrule::R
end

function DI.prepare_pullback(f, ::AutoTapir, x)
    y = f(x)
    rrule = build_rrule(f, x)
    return TapirOneArgPullbackExtras(y, rrule)
end

function DI.value_and_pullback(
    f, ::AutoTapir, x, dy, extras::TapirOneArgPullbackExtras{Y}
) where {Y}
    dy_righttype = convert(tangent_type(Y), dy)
    new_y, (new_df, new_dx) = value_and_pullback!!(extras.rrule, dy_righttype, f, x)
    return new_y, new_dx
end

function DI.value_and_pullback!(
    f, dx, ::AutoTapir, x, dy, extras::TapirOneArgPullbackExtras{Y}
) where {Y}
    dy_righttype = convert(tangent_type(Y), dy)
    dx_righttype = convert(tangent_type(typeof(x)), dx)
    dx_righttype = set_to_zero!!(dx_righttype)
    new_y, (new_df, new_dx) = value_and_pullback!!(
        extras.rrule, dy_righttype, zero_codual(f), CoDual(x, dx_righttype)
    )
    return new_y, new_dx
end

function DI.pullback(f, backend::AutoTapir, x, dy, extras::TapirOneArgPullbackExtras)
    return DI.value_and_pullback(f, backend, x, dy, extras)[2]
end

function DI.pullback!(
    f, dx, backend::AutoTapir, x, dy, extras::TapirOneArgPullbackExtras
)
    return DI.value_and_pullback!(f, dx, backend, x, dy, extras)[2]
end
