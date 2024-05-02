## Docstrings

"""
    prepare_pullback(f,     backend, x, dy) -> extras
    prepare_pullback(f!, y, backend, x, dy) -> extras

Create an `extras` object subtyping [`PullbackExtras`](@ref) that can be given to pullback operators.

Beware that in the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_pullback end

"""
    value_and_pullback(f,     backend, x, dy, [extras]) -> (y, dx)
    value_and_pullback(f!, y, backend, x, dy, [extras]) -> (y, dx)

!!! info
    Required primitive for reverse mode backends.
"""
function value_and_pullback end

"""
    value_and_pullback!(f,     dx, backend, x, dy, [extras]) -> (y, dx)
    value_and_pullback!(f!, y, dx, backend, x, dy, [extras]) -> (y, dx)
"""
function value_and_pullback! end

"""
    pullback(f,     backend, x, dy, [extras]) -> dx
    pullback(f!, y, backend, x, dy, [extras]) -> dx
"""
function pullback end

"""
    pullback!(f,     dx, backend, x, dy, [extras]) -> dx
    pullback!(f!, y, dx, backend, x, dy, [extras]) -> dx
"""
function pullback! end

"""
    value_and_pullback_split(f,     backend, x, [extras]) -> (y, pbf(   dy) -> dx)
    value_and_pullback_split(f!, y, backend, x, [extras]) -> (y, pbf(y, dy) -> dx)
"""
function value_and_pullback_split end

"""
    value_and_pullback!_split(f,     backend, x, [extras]) -> (y, pbf!(   dx, dy) -> dx)
    value_and_pullback!_split(f!, y, backend, x, [extras]) -> (y, pbf!(y, dx, dy) -> dx)
"""
function value_and_pullback!_split end

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

function prepare_pullback(f::F, backend::AbstractADType, x, dy) where {F}
    return prepare_pullback_aux(f, backend, x, dy, pullback_performance(backend))
end

function prepare_pullback(f!::F, y, backend::AbstractADType, x, dy) where {F}
    return prepare_pullback_aux(f!, y, backend, x, dy, pullback_performance(backend))
end

function prepare_pullback_aux(f::F, backend, x, dy, ::PullbackSlow) where {F}
    dx = x isa Number ? one(x) : basis(backend, x, first(CartesianIndices(x)))
    pushforward_extras = prepare_pushforward(f, backend, x, dx)
    return PushforwardPullbackExtras(pushforward_extras)
end

function prepare_pullback_aux(f!::F, y, backend, x, dy, ::PullbackSlow) where {F}
    dx = x isa Number ? one(x) : basis(backend, x, first(CartesianIndices(x)))
    pushforward_extras = prepare_pushforward(f!, y, backend, x, dx)
    return PushforwardPullbackExtras(pushforward_extras)
end

# Throw error if backend is missing
prepare_pullback_aux(f::F, backend, x, dy, ::PullbackFast) where {F}     = throw(MissingBackendError(backend))
prepare_pullback_aux(f!::F, y, backend, x, dy, ::PullbackFast) where {F} = throw(MissingBackendError(backend))

## One argument

function value_and_pullback(
    f::F,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x, dy),
) where {F}
    return value_and_pullback_onearg_aux(f, backend, x, dy, extras)
end

function value_and_pullback_onearg_aux(
    f::F, backend, x, dy, extras::PushforwardPullbackExtras
) where {F}
    @compat (; pushforward_extras) = extras
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

function value_and_pullback!(
    f::F,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x, dy),
) where {F}
    y, new_dx = value_and_pullback(f, backend, x, dy, extras)
    return y, copyto!(dx, new_dx)
end

function pullback(
    f::F,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x, dy),
) where {F}
    return value_and_pullback(f, backend, x, dy, extras)[2]
end

function pullback!(
    f::F,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f, backend, x, dy),
) where {F}
    return value_and_pullback!(f, dx, backend, x, dy, extras)[2]
end

## Two arguments

function value_and_pullback(
    f!::F,
    y,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f!, y, backend, x, dy),
) where {F}
    return value_and_pullback_twoarg_aux(f!, y, backend, x, dy, extras)
end

function value_and_pullback_twoarg_aux(
    f!::F, y, backend, x, dy, extras::PushforwardPullbackExtras
) where {F}
    @compat (; pushforward_extras) = extras
    dx = if x isa Number && y isa AbstractArray
        dot(dy, pushforward(f!, y, backend, x, one(x), pushforward_extras))
    elseif x isa AbstractArray && y isa AbstractArray
        map(CartesianIndices(x)) do j
            dot(dy, pushforward(f!, y, backend, x, basis(backend, x, j), pushforward_extras))
        end
    end
    f!(y, x)
    return y, dx
end

function value_and_pullback!(
    f!::F,
    y,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f!, y, backend, x, dy),
) where {F}
    y, new_dx = value_and_pullback(f!, y, backend, x, dy, extras)
    return y, copyto!(dx, new_dx)
end

function pullback(
    f!::F,
    y,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f!, y, backend, x, dy),
) where {F}
    return value_and_pullback(f!, y, backend, x, dy, extras)[2]
end

function pullback!(
    f!::F,
    y,
    dx,
    backend::AbstractADType,
    x,
    dy,
    extras::PullbackExtras=prepare_pullback(f!, y, backend, x, dy),
) where {F}
    return value_and_pullback!(f!, y, dx, backend, x, dy, extras)[2]
end

## Split one argument

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

function (pbf::OneArgPullbackFunc)(dy)
    @compat (; f, backend, x, extras) = pbf
    return pullback(f, backend, x, dy, extras)
end

function (pbf::OneArgPullbackFunc!)(dx, dy)
    @compat (; f, backend, x, extras) = pbf
    return pullback!(f, dx, backend, x, dy, extras)
end

function value_and_pullback_split(
    f::F,
    backend::AbstractADType,
    x,
    extras::PullbackExtras=prepare_pullback(f, backend, x, f(x)),
) where {F}
    return f(x), OneArgPullbackFunc(f, backend, x, extras)
end

function value_and_pullback!_split(
    f::F,
    backend::AbstractADType,
    x,
    extras::PullbackExtras=prepare_pullback(f, backend, x, f(x)),
) where {F}
    return f(x), OneArgPullbackFunc!(f, backend, x, extras)
end

## Split two argument

struct TwoArgPullbackFunc{B,F,X,E}
    f!::F
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

function (pbf::TwoArgPullbackFunc)(y, dy)
    @compat (; f!, backend, x, extras) = pbf
    return pullback(f!, y, backend, x, dy, extras)
end

function (pbf::TwoArgPullbackFunc!)(y, dx, dy)
    @compat (; f!, backend, x, extras) = pbf
    return pullback!(f!, y, dx, backend, x, dy, extras)
end

function value_and_pullback_split(
    f!::F,
    y,
    backend::AbstractADType,
    x,
    extras::PullbackExtras=prepare_pullback(f!, y, backend, x, similar(y)),
) where {F}
    f!(y, x)
    return y, TwoArgPullbackFunc(f!, backend, x, extras)
end

function value_and_pullback!_split(
    f!::F,
    y,
    backend::AbstractADType,
    x,
    extras::PullbackExtras=prepare_pullback(f!, y, backend, x, similar(y)),
) where {F}
    f!(y, x)
    return y, TwoArgPullbackFunc!(f!, backend, x, extras)
end
