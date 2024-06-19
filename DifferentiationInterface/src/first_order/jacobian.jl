## Docstrings

"""
    prepare_jacobian(f,     backend, x) -> extras
    prepare_jacobian(f!, y, backend, x) -> extras

Create an `extras` object that can be given to [`jacobian`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    In the two-argument case, `y` is mutated by `f!` during preparation.
"""
function prepare_jacobian end

"""
    value_and_jacobian(f,     backend, x, [extras]) -> (y, jac)
    value_and_jacobian(f!, y, backend, x, [extras]) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`.
"""
function value_and_jacobian end

"""
    value_and_jacobian!(f,     jac, backend, x, [extras]) -> (y, jac)
    value_and_jacobian!(f!, y, jac, backend, x, [extras]) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.
"""
function value_and_jacobian! end

"""
    jacobian(f,     backend, x, [extras]) -> jac
    jacobian(f!, y, backend, x, [extras]) -> jac

Compute the Jacobian matrix of the function `f` at point `x`.
"""
function jacobian end

"""
    jacobian!(f,     jac, backend, x, [extras]) -> jac
    jacobian!(f!, y, jac, backend, x, [extras]) -> jac

Compute the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.
"""
function jacobian! end

## Preparation

"""
    JacobianExtras

Abstract type for additional information needed by [`jacobian`](@ref) and its variants.
"""
abstract type JacobianExtras <: Extras end

struct NoJacobianExtras <: JacobianExtras end

struct PushforwardJacobianExtras{B,D,E<:PushforwardExtras} <: JacobianExtras
    seeds::D
    pushforward_batched_extras::E
end

struct PullbackJacobianExtras{B,D,E<:PullbackExtras,Y} <: JacobianExtras
    seeds::D
    pullback_batched_extras::E
    y_example::Y
end

function prepare_jacobian(f::F, backend::AbstractADType, x) where {F}
    y = f(x)
    return prepare_jacobian_aux((f,), backend, x, y, pushforward_performance(backend))
end

function prepare_jacobian(f!::F, y, backend::AbstractADType, x) where {F}
    return prepare_jacobian_aux((f!, y), backend, x, y, pushforward_performance(backend))
end

function prepare_jacobian_aux(f_or_f!y::FY, backend, x, y, ::PushforwardFast) where {FY}
    N = length(x)
    B = pick_batchsize(backend, N)
    seeds = [basis(backend, x, ind) for ind in CartesianIndices(x)]
    pushforward_batched_extras = prepare_pushforward_batched(
        f_or_f!y..., backend, x, Batch(ntuple(Returns(seeds[1]), Val(B)))
    )
    D = typeof(seeds)
    E = typeof(pushforward_batched_extras)
    return PushforwardJacobianExtras{B,D,E}(seeds, pushforward_batched_extras)
end

function prepare_jacobian_aux(f_or_f!y::FY, backend, x, y, ::PushforwardSlow) where {FY}
    M = length(y)
    B = pick_batchsize(backend, M)
    seeds = [basis(backend, y, ind) for ind in CartesianIndices(y)]
    pullback_batched_extras = prepare_pullback_batched(
        f_or_f!y..., backend, x, Batch(ntuple(Returns(seeds[1]), Val(B)))
    )
    D = typeof(seeds)
    E = typeof(pullback_batched_extras)
    Y = typeof(y)
    return PullbackJacobianExtras{B,D,E,Y}(seeds, pullback_batched_extras, copy(y))
end

## One argument

function jacobian(f::F, backend::AbstractADType, x) where {F}
    return jacobian(f, backend, x, prepare_jacobian(f, backend, x))
end

function jacobian!(f::F, jac, backend::AbstractADType, x) where {F}
    return jacobian!(f, jac, backend, x, prepare_jacobian(f, backend, x))
end

function value_and_jacobian(f::F, backend::AbstractADType, x) where {F}
    return value_and_jacobian(f, backend, x, prepare_jacobian(f, backend, x))
end

function value_and_jacobian!(f::F, jac, backend::AbstractADType, x) where {F}
    return value_and_jacobian!(f, jac, backend, x, prepare_jacobian(f, backend, x))
end

function jacobian(f::F, backend::AbstractADType, x, extras::JacobianExtras) where {F}
    return jacobian_aux((f,), backend, x, extras)
end

function jacobian!(f::F, jac, backend::AbstractADType, x, extras::JacobianExtras) where {F}
    return jacobian_aux!((f,), jac, backend, x, extras)
end

function value_and_jacobian(
    f::F, backend::AbstractADType, x, extras::JacobianExtras
) where {F}
    return f(x), jacobian(f, backend, x, extras)
end

function value_and_jacobian!(
    f::F, jac, backend::AbstractADType, x, extras::JacobianExtras
) where {F}
    return f(x), jacobian!(f, jac, backend, x, extras)
end

## Two arguments

function jacobian(f!::F, y, backend::AbstractADType, x) where {F}
    return jacobian(f!, y, backend, x, prepare_jacobian(f!, y, backend, x))
end

function jacobian!(f!::F, y, jac, backend::AbstractADType, x) where {F}
    return jacobian!(f!, y, jac, backend, x, prepare_jacobian(f!, y, backend, x))
end

function value_and_jacobian(f!::F, y, backend::AbstractADType, x) where {F}
    return value_and_jacobian(f!, y, backend, x, prepare_jacobian(f!, y, backend, x))
end

function value_and_jacobian!(f!::F, y, jac, backend::AbstractADType, x) where {F}
    return value_and_jacobian!(f!, y, jac, backend, x, prepare_jacobian(f!, y, backend, x))
end

function jacobian(f!::F, y, backend::AbstractADType, x, extras::JacobianExtras) where {F}
    return jacobian_aux((f!, y), backend, x, extras)
end

function jacobian!(
    f!::F, y, jac, backend::AbstractADType, x, extras::JacobianExtras
) where {F}
    return jacobian_aux!((f!, y), jac, backend, x, extras)
end

function value_and_jacobian(
    f!::F, y, backend::AbstractADType, x, extras::JacobianExtras
) where {F}
    jac = jacobian(f!, y, backend, x, extras)
    f!(y, x)
    return y, jac
end

function value_and_jacobian!(
    f!::F, y, jac, backend::AbstractADType, x, extras::JacobianExtras
) where {F}
    jacobian!(f!, y, jac, backend, x, extras)
    f!(y, x)
    return y, jac
end

## Common auxiliaries

function jacobian_aux(
    f_or_f!y::FY, backend, x::AbstractArray, extras::PushforwardJacobianExtras{B}
) where {FY,B}
    @compat (; seeds, pushforward_batched_extras) = extras
    N = length(x)

    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f_or_f!y...,
        backend,
        x,
        Batch(ntuple(Returns(seeds[1]), Val(B))),
        pushforward_batched_extras,
    )

    jac = mapreduce(hcat, 1:div(N, B, RoundUp)) do a
        dx_batch_elements = ntuple(Val(B)) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % N]
        end
        dy_batch = pushforward_batched(
            f_or_f!y...,
            backend,
            x,
            Batch(dx_batch_elements),
            pushforward_batched_extras_same,
        )
        stack(vec, dy_batch.elements; dims=2)
    end

    if N < size(jac, 2)
        jac = jac[:, 1:N]
    end
    return jac
end

function jacobian_aux(
    f_or_f!y::FY, backend, x::AbstractArray, extras::PullbackJacobianExtras{B}
) where {FY,B}
    @compat (; seeds, pullback_batched_extras, y_example) = extras
    M = length(y_example)

    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f_or_f!y...,
        backend,
        x,
        Batch(ntuple(Returns(seeds[1]), Val(B))),
        extras.pullback_batched_extras,
    )

    jac = mapreduce(vcat, 1:div(M, B, RoundUp)) do a
        dy_batch_elements = ntuple(Val(B)) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % M]
        end
        dx_batch = pullback_batched(
            f_or_f!y...,
            backend,
            x,
            Batch(dy_batch_elements),
            pullback_batched_extras_same,
        )
        stack(vec, dx_batch.elements; dims=1)
    end

    if M < size(jac, 1)
        jac = jac[1:M, :]
    end
    return jac
end

function jacobian_aux!(
    f_or_f!y::FY,
    jac::AbstractMatrix,
    backend,
    x::AbstractArray,
    extras::PushforwardJacobianExtras{B},
) where {FY,B}
    @compat (; seeds, pushforward_batched_extras) = extras
    N = length(x)

    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f_or_f!y...,
        backend,
        x,
        Batch(ntuple(Returns(seeds[1]), Val(B))),
        pushforward_batched_extras,
    )

    for a in 1:div(N, B, RoundUp)
        dx_batch_elements = ntuple(Val(B)) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % N]
        end
        dy_batch_elements = ntuple(Val(B)) do b
            reshape(view(jac, :, 1 + ((a - 1) * B + (b - 1)) % N), size(y))
        end
        pushforward_batched!(
            f_or_f!y...,
            Batch(dy_batch_elements),
            backend,
            x,
            Batch(dx_batch_elements),
            pushforward_batched_extras_same,
        )
    end

    return jac
end

function jacobian_aux!(
    f_or_f!y::FY,
    jac::AbstractMatrix,
    backend,
    x::AbstractArray,
    extras::PullbackJacobianExtras{B},
) where {FY,B}
    @compat (; seeds, pullback_batched_extras, y_example) = extras
    M = length(y_example)

    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f_or_f!y...,
        backend,
        x,
        Batch(ntuple(Returns(seeds[1]), Val(B))),
        extras.pullback_batched_extras,
    )

    for a in 1:div(M, B, RoundUp)
        dy_batch_elements = ntuple(Val(B)) do b
            seeds[1 + ((a - 1) * B + (b - 1)) % M]
        end
        dx_batch_elements = ntuple(Val(B)) do b
            reshape(view(jac, 1 + ((a - 1) * B + (b - 1)) % M, :), size(x))
        end
        pullback_batched!(
            f_or_f!y...,
            Batch(dx_batch_elements),
            backend,
            x,
            Batch(dy_batch_elements),
            pullback_batched_extras_same,
        )
    end

    return jac
end
