struct TapirMutatingPullbackExtras{R} <: PullbackExtras
    rrule::R
end

function DI.prepare_pullback(f!, ::AutoTapir, y, x)
    return TapirMutatingPullbackExtras(build_rrule(f!, y, x))
end

function DI.value_and_pullback!!(
    f!, y, dx, ::AutoTapir, x, dy, extras::TapirMutatingPullbackExtras
)
    dy_righttype = convert(tangent_type(typeof(y)), dy)
    dx_righttype = convert(tangent_type(typeof(x)), dx)
    dx_righttype = zero!!(dx_righttype)
    @info "before" x y dx_righttype dy_righttype
    new_y, (new_df!, new_dy, new_dx) = value_and_pullback!!(
        extras.rrule,
        NoTangent(),
        zero_codual(f!),
        CoDual(y, dy_righttype),
        CoDual(x, dx_righttype),
    )
    @info "after" x y dx_righttype dy_righttype
    @info "result" new_y new_dx new_dy
    return y, new_dx
end
