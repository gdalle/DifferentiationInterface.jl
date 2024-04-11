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
    return copyto!(dy, value_and_pullback(f, backend, x, dy, extras))
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
    value_and_pullback!(f!, y, dx, backend, x, dy, [extras]) -> (y, dx)

!!! info
    Required primitive for reverse mode backends to support mutating functions.
"""
function value_and_pullback!(
    f!,
    y,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f!, backend, y, x),
)
    return value_and_pullback_aux!(f!, y, dx, backend, x, dy, extras)
end

function value_and_pullback_aux!(
    f!, y, dx, backend, x, dy, extras::PushforwardPullbackExtras
)
    (; pushforward_extras) = extras
    if x isa AbstractArray && y isa AbstractArray
        map!(dx, CartesianIndices(x)) do j
            dot(
                dy,
                pushforward!(
                    f!,
                    y,
                    similar(y),
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
    pullback!(f!, y, dx, backend, x, dy, [extras]) -> dx

"""
function pullback!(
    f!,
    y,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f!, backend, y, x),
)
    return value_and_pullback!(f!, y, dx, backend, x, dy, extras)[2]
end

## Split

struct AllocatingPullbackFunc{B,F,X,E}
    f::F
    backend::B
    x::X
    extras::E
end

struct AllocatingPullbackFunc!!{B,F,X,E}
    f::F
    backend::B
    x::X
    extras::E
end

struct MutatingPullbackFunc!!{B,F,X,E}
    f!::F
    backend::B
    x::X
    extras::E
end

function (pbf::AllocatingPullbackFunc)(dy)
    (; f, backend, x, extras) = pbf
    return pullback(f, backend, x, dy, extras)
end

function (pbf::AllocatingPullbackFunc!!)(dx, dy)
    (; f, backend, x, extras) = pbf
    return pullback!(f, dx, backend, x, dy, extras)
end

function (pbf::MutatingPullbackFunc!!)(y, dx, dy)
    (; f!, backend, x, extras) = pbf
    return pullback!(f!, y, dx, backend, x, dy, extras)
end

"""
    value_and_pullback_split(f, backend, x, [extras]) -> (y, pullbackfunc(dy) -> dx)
"""
function value_and_pullback_split(
    f, backend::AbstractADType, x, extras::PullbackExtras=prepare_pullback(f, backend, x)
)
    return f(x), AllocatingPullbackFunc(f, backend, x, extras)
end

"""
    value_and_pullback!_split(f, backend, x, [extras]) -> (y, pullbackfunc!(dx, dy) -> dx)
"""
function value_and_pullback!_split(
    f, backend::AbstractADType, x, extras::PullbackExtras=prepare_pullback(f, backend, x)
)
    return f(x), AllocatingPullbackFunc!(f, backend, x, extras)
end

"""
    value_and_pullback!_split!(f!, y, backend, x, [extras]) -> (y, pullbackfunc!(y, dx, dy) -> dx)
"""
function value_and_pullback!_split!(
    f!,
    y,
    backend::AbstractADType,
    x,
    extras::PullbackExtras=prepare_pullback(f!, backend, y, x),
)
    f!(y, x)
    return y, MutatingPullbackFunc!(f!, backend, x, extras)
end
