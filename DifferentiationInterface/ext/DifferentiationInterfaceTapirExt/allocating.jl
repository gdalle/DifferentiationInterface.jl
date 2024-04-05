struct TapirAllocatingPullbackExtras{R} <: PullbackExtras
    rrule::R
end

DI.prepare_pullback(f, ::AutoTapir, x) = TapirAllocatingPullbackExtras(build_rrule(f, x))

function DI.value_and_pullback(f, ::AutoTapir, x, dy, extras::TapirAllocatingPullbackExtras)
    y = f(x)  # TODO: one call too many, just for the conversion
    dy_righttype = convert(tangent_type(typeof(y)), dy)
    new_y, (new_df, new_dx) = value_and_pullback!!(extras.rrule, dy_righttype, f, x)
    return new_y, new_dx
end

function DI.value_and_pullback!!(
    f, dx, ::AutoTapir, x, dy, extras::TapirAllocatingPullbackExtras
)
    y = f(x)  # TODO: one call too many, just for the conversion
    dy_righttype = convert(tangent_type(typeof(y)), dy)
    dx_righttype = convert(tangent_type(typeof(x)), dx)
    dx_righttype = set_to_zero!!(dx_righttype)
    new_y, (new_df, new_dx) = value_and_pullback!!(
        extras.rrule, dy_righttype, zero_codual(f), CoDual(x, dx_righttype)
    )
    return new_y, new_dx
end
