## Preparation

"""
    PullbackExtras

Abstract type for additional information needed by pullback operators.
"""
abstract type PullbackExtras <: Extras end

struct NoPullbackExtras <: PullbackExtras end

struct PushforwardPullbackExtras{E} <: PullbackExtras
    pushforward_extras::E
end

"""
    prepare_pullback(f, backend, x) -> extras
    prepare_pullback(f!, backend, y, x) -> extras

Create an `extras` object subtyping [`PullbackExtras`](@ref) that can be given to pullback operators.
"""
function prepare_pullback(f, backend::AbstractADType, x)
    return prepare_pullback_aux(f, backend, x, pullback_performance(backend))
end

function prepare_pullback(f!, backend::AbstractADType, y, x)
    return prepare_pullback_aux(f!, backend, y, x, pullback_performance(backend))
end

function prepare_pullback_aux(f, backend, x, ::PullbackSlow)
    return PushforwardPullbackExtras(prepare_pushforward(f, backend, x))
end

function prepare_pullback_aux(f!, backend, y, x, ::PullbackSlow)
    return PushforwardPullbackExtras(prepare_pushforward(f!, backend, y, x))
end

## One argument

"""
    value_and_pullback(f, backend, x, dy, [extras]) -> (y, dx)

!!! info
    Required primitive for reverse mode backends to support allocating functions.
"""
function value_and_pullback(
    f,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x),
)
    return value_and_pullback_aux(f, backend, x, dy, extras)
end

function value_and_pullback_aux(f, backend, x, dy, extras::PushforwardPullbackExtras)
    (; pushforward_extras) = extras
    y = f(x)
    dx = if x isa Number && y isa Number
        dy * pushforward(f, backend, x, one(x), pushforward_extras)
    elseif x isa Number && y isa AbstractArray
        dot(dy, pushforward(f, backend, x, one(x), pushforward_extras))
    elseif x isa AbstractArray && y isa Number
        map(CartesianIndices(x)) do j
            dy * pushforward(f, backend, x, basis(backend, x, j), pushforward_extras)
        end
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(x)) do j
            dot(dy, pushforward(f, backend, x, basis(backend, x, j), pushforward_extras))
        end
    end
    return y, dx
end

"""
    value_and_pullback!(f, dx, backend, x, dy, [extras]) -> (y, dx)
"""
function value_and_pullback!(
    f,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x),
)
    y, new_dx = value_and_pullback(f, backend, x, dy, extras)
    return y, copyto!(dx, new_dx)
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
    return value_and_pullback(f, backend, x, dy, extras)[2]
end

"""
    pullback!(f, dx, backend, x, dy, [extras]) -> dx
"""
function pullback!(
    f,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x),
)
    return value_and_pullback!(f, dx, backend, x, dy, extras)[2]
end

## Two arguments

"""
    value_and_pullback!(f!, (y, dx), backend, x, dy, [extras]) -> (y, dx)

!!! info
    Required primitive for reverse mode backends to support mutating functions.
"""
function value_and_pullback!(
    f!,
    y_and_dx::Tuple,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f!, backend, y_and_dx[1], x),
)
    return value_and_pullback_twoarg_aux!(f!, y_and_dx, backend, x, dy, extras)
end

function value_and_pullback_twoarg_aux!(
    f!, (y, dx), backend, x, dy, extras::PushforwardPullbackExtras
)
    (; pushforward_extras) = extras
    if x isa Number && y isa AbstractArray
        dx = dot(
            dy, pushforward!(f!, (y, similar(y)), backend, x, one(x), pushforward_extras)
        )
    elseif x isa AbstractArray && y isa AbstractArray
        map!(dx, CartesianIndices(x)) do j
            dot(
                dy,
                pushforward!(
                    f!,
                    (y, similar(y)),
                    backend,
                    x,
                    basis(backend, x, j),
                    pushforward_extras,
                ),
            )
        end
    end
    f!(y, x)
    return y, dx
end

"""
    pullback!(f!, (y, dx), backend, x, dy, [extras]) -> dx

"""
function pullback!(
    f!,
    y_and_dx::Tuple,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f!, backend, y_and_dx[1], x),
)
    return value_and_pullback!(f!, y_and_dx, backend, x, dy, extras)[2]
end

## Split

struct OneArgPullbackFunc{B,F,X,E}
    f::F
    backend::B
    x::X
    extras::E
end

struct OneArgPullbackFunc!{B,F,X,E}
    f::F
    backend::B
    x::X
    extras::E
end

struct TwoArgPullbackFunc!{B,F,X,E}
    f!::F
    backend::B
    x::X
    extras::E
end

function (pbf::OneArgPullbackFunc)(dy)
    (; f, backend, x, extras) = pbf
    return pullback(f, backend, x, dy, extras)
end

function (pbf::OneArgPullbackFunc!)(dx, dy)
    (; f, backend, x, extras) = pbf
    return pullback!(f, dx, backend, x, dy, extras)
end

function (pbf::TwoArgPullbackFunc!)((y, dx)::Tuple, dy)
    (; f!, backend, x, extras) = pbf
    return pullback!(f!, (y, dx), backend, x, dy, extras)
end

"""
    value_and_pullback_split(f, backend, x, [extras]) ->
        (y, pullbackfunc(dy) -> dx)
"""
function value_and_pullback_split(
    f, backend::AbstractADType, x, extras::PullbackExtras=prepare_pullback(f, backend, x)
)
    return f(x), OneArgPullbackFunc(f, backend, x, extras)
end

"""
    value_and_pullback!_split(f, backend, x, [extras]) ->
        (y, pullbackfunc!(dx, dy) -> dx)
"""
function value_and_pullback!_split(
    f, backend::AbstractADType, x, extras::PullbackExtras=prepare_pullback(f, backend, x)
)
    return f(x), OneArgPullbackFunc!(f, backend, x, extras)
end

"""
    value_and_pullback!_split!(f!, y, backend, x, [extras]) ->
        (y, pullbackfunc!((y, dx), dy) -> dx)
"""
function value_and_pullback!_split!(
    f!,
    y,
    backend::AbstractADType,
    x,
    extras::PullbackExtras=prepare_pullback(f!, backend, y, x),
)
    f!(y, x)
    return y, TwoArgPullbackFunc!(f!, backend, x, extras)
end
