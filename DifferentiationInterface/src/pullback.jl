## Preparation

"""
    PullbackExtras

Abstract type for additional information needed by pullback operators.
"""
abstract type PullbackExtras <: Extras end

struct NoPullbackExtras <: PullbackExtras end

"""
    prepare_pullback(f, backend, x) -> extras
    prepare_pullback(f!, backend, y, x) -> extras

Create an `extras` object subtyping [`PullbackExtras`](@ref) that can be given to pullback operators.
"""
prepare_pullback(f, ::AbstractADType, x) = NoPullbackExtras()
prepare_pullback(f!, ::AbstractADType, y, x) = NoPullbackExtras()

## Closure

"""
    value_and_pullback_split(f, backend, x, [extras]) -> (y, pullbackfunc(dy) -> dx)

!!! info
    Required primitive for reverse mode backends to support allocating functions.
"""
function value_and_pullback_split(
    f, backend::AbstractADType, x, extras::PullbackExtras=prepare_pullback(f, backend, x)
)
    return value_and_pullback_split_aux(
        f, backend, x, extras, pullback_performance(backend)
    )
end

function value_and_pullback_split_aux(f, backend, x, extras, ::PullbackSlow)
    pushforward_extras = prepare_pushforward(f, backend, x)
    y = f(x)
    pullbackfunc = if x isa Number && y isa Number
        dy -> dy * pushforward(f, backend, x, one(x), pushforward_extras)
    elseif x isa Number && y isa AbstractArray
        dy -> dot(dy, pushforward(f, backend, x, one(x), pushforward_extras))
    elseif x isa AbstractArray && y isa Number
        dy -> map(CartesianIndices(x)) do j
            dy * pushforward(f, backend, x, basis(backend, x, j), pushforward_extras)
        end
    elseif x isa AbstractArray && y isa AbstractArray
        dy -> map(CartesianIndices(x)) do j
            dot(dy, pushforward(f, backend, x, basis(backend, x, j), pushforward_extras))
        end
    end
    return y, pullbackfunc
end

"""
    value_and_pullback!!_split(f, backend, x, [extras]) -> (y, pullbackfunc!!(dx, dy) -> dx)
"""
function value_and_pullback!!_split(
    f, backend::AbstractADType, x, extras::PullbackExtras=prepare_pullback(f, backend, x)
)
    y, pullbackfunc = value_and_pullback_split(f, backend, x, extras)
    pullbackfunc!!(_dx, dy) = pullbackfunc(dy)
    return y, pullbackfunc!!
end

## Allocating

"""
    value_and_pullback(f, backend, x, dy, [extras]) -> (y, dx)
"""
function value_and_pullback(
    f,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x),
)
    y, pullbackfunc = value_and_pullback_split(f, backend, x, extras)
    return y, pullbackfunc(dy)
end

"""
    value_and_pullback!!(f, dx, backend, x, dy, [extras]) -> (y, dx)
"""
function value_and_pullback!!(
    f,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x),
)
    y, pullbackfunc!! = value_and_pullback!!_split(f, backend, x, extras)
    return y, pullbackfunc!!(dx, dy)
end

"""
    pullback(f, backend, x, dy, [extras]) -> dx
"""
function pullback(
    f,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x),
)
    _, pullbackfunc = value_and_pullback_split(f, backend, x, extras)
    return pullbackfunc(dy)
end

"""
    pullback!!(f, dx, backend, x, dy, [extras]) -> dx
"""
function pullback!!(
    f,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x),
)
    _, pullbackfunc!! = value_and_pullback!!_split(f, backend, x, extras)
    return pullbackfunc!!(dx, dy)
end

## Mutating

"""
    value_and_pullback!!(f!, y, dx, backend, x, dy, [extras]) -> (y, dx)

!!! info
    Required primitive for reverse mode backends to support mutating functions.
"""
function value_and_pullback!!(
    f!,
    y,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f!, backend, y, x),
)
    new_extras = prepare_pushforward(f!, backend, y, x)
    dx = if x isa Number && y isa AbstractArray
        dot(dy, value_and_pushforward!!(f!, y, similar(y), backend, x, one(x), new_extras)[2])
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(x)) do j
            dot(
                dy,
                value_and_pushforward!!(
                    f!, y, similar(y), backend, x, basis(backend, x, j), new_extras
                )[2],
            )
        end
    end
    return y, dx
end
