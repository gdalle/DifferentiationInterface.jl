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

$(document_preparation("jacobian"))
"""
function value_and_jacobian end

"""
    value_and_jacobian!(f,     jac, backend, x, [extras]) -> (y, jac)
    value_and_jacobian!(f!, y, jac, backend, x, [extras]) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.
    
$(document_preparation("jacobian"))
"""
function value_and_jacobian! end

"""
    jacobian(f,     backend, x, [extras]) -> jac
    jacobian(f!, y, backend, x, [extras]) -> jac

Compute the Jacobian matrix of the function `f` at point `x`.

$(document_preparation("jacobian"))
"""
function jacobian end

"""
    jacobian!(f,     jac, backend, x, [extras]) -> jac
    jacobian!(f!, y, jac, backend, x, [extras]) -> jac

Compute the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.

$(document_preparation("jacobian"))
"""
function jacobian! end

## Preparation

"""
    JacobianExtras

Abstract type for additional information needed by [`jacobian`](@ref) and its variants.
"""
abstract type JacobianExtras <: Extras end

struct NoJacobianExtras <: JacobianExtras end

struct PushforwardJacobianExtras{B,D,R,E<:PushforwardExtras} <: JacobianExtras
    batched_seeds::Vector{Tangents{B,D}}
    batched_results::Vector{Tangents{B,R}}
    pushforward_extras::E
    N::Int
end

struct PullbackJacobianExtras{B,D,R,E<:PullbackExtras} <: JacobianExtras
    batched_seeds::Vector{Tangents{B,D}}
    batched_results::Vector{Tangents{B,R}}
    pullback_extras::E
    M::Int
end

function prepare_jacobian(f::F, backend::AbstractADType, x) where {F}
    y = f(x)
    return prepare_jacobian_aux((f,), backend, x, y, pushforward_performance(backend))
end

function prepare_jacobian(f!::F, y, backend::AbstractADType, x) where {F}
    return prepare_jacobian_aux((f!, y), backend, x, y, pushforward_performance(backend))
end

function prepare_jacobian_aux(
    f_or_f!y::FY, backend::AbstractADType, x, y, ::PushforwardFast
) where {FY}
    N = length(x)
    B = pick_batchsize(backend, N)
    seeds = [basis(backend, x, ind) for ind in CartesianIndices(x)]
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B))...) for
        a in 1:div(N, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(y), Val(B))...) for _ in batched_seeds]
    pushforward_extras = prepare_pushforward(f_or_f!y..., backend, x, batched_seeds[1])
    D = eltype(batched_seeds[1])
    R = eltype(batched_results[1])
    E = typeof(pushforward_extras)
    return PushforwardJacobianExtras{B,D,R,E}(
        batched_seeds, batched_results, pushforward_extras, N
    )
end

function prepare_jacobian_aux(
    f_or_f!y::FY, backend::AbstractADType, x, y, ::PushforwardSlow
) where {FY}
    M = length(y)
    B = pick_batchsize(backend, M)
    seeds = [basis(backend, y, ind) for ind in CartesianIndices(y)]
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % M], Val(B))...) for
        a in 1:div(M, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(x), Val(B))...) for _ in batched_seeds]
    pullback_extras = prepare_pullback(f_or_f!y..., backend, x, batched_seeds[1])
    D = eltype(batched_seeds[1])
    R = eltype(batched_results[1])
    E = typeof(pullback_extras)
    return PullbackJacobianExtras{B,D,R,E}(
        batched_seeds, batched_results, pullback_extras, M
    )
end

## One argument

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
    f_or_f!y::FY, backend::AbstractADType, x, extras::PushforwardJacobianExtras{B}
) where {FY,B}
    @compat (; batched_seeds, pushforward_extras, N) = extras

    pushforward_extras_same = prepare_pushforward_same_point(
        f_or_f!y..., backend, x, batched_seeds[1], pushforward_extras
    )

    jac_blocks = map(eachindex(batched_seeds)) do a
        dy_batch = pushforward(
            f_or_f!y..., backend, x, batched_seeds[a], pushforward_extras_same
        )
        stack(vec, dy_batch.d; dims=2)
    end

    jac = reduce(hcat, jac_blocks)
    if N < size(jac, 2)
        jac = jac[:, 1:N]
    end
    return jac
end

function jacobian_aux(
    f_or_f!y::FY, backend::AbstractADType, x, extras::PullbackJacobianExtras{B}
) where {FY,B}
    @compat (; batched_seeds, pullback_extras, M) = extras

    pullback_extras_same = prepare_pullback_same_point(
        f_or_f!y..., backend, x, batched_seeds[1], extras.pullback_extras
    )

    jac_blocks = map(eachindex(batched_seeds)) do a
        dx_batch = pullback(f_or_f!y..., backend, x, batched_seeds[a], pullback_extras_same)
        stack(vec, dx_batch.d; dims=1)
    end

    jac = reduce(vcat, jac_blocks)
    if M < size(jac, 1)
        jac = jac[1:M, :]
    end
    return jac
end

function jacobian_aux!(
    f_or_f!y::FY, jac, backend::AbstractADType, x, extras::PushforwardJacobianExtras{B}
) where {FY,B}
    @compat (; batched_seeds, batched_results, pushforward_extras, N) = extras

    pushforward_extras_same = prepare_pushforward_same_point(
        f_or_f!y..., backend, x, batched_seeds[1], pushforward_extras
    )

    for a in eachindex(batched_seeds, batched_results)
        pushforward!(
            f_or_f!y...,
            batched_results[a],
            backend,
            x,
            batched_seeds[a],
            pushforward_extras_same,
        )

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(jac, :, 1 + ((a - 1) * B + (b - 1)) % N), vec(batched_results[a].d[b])
            )
        end
    end

    return jac
end

function jacobian_aux!(
    f_or_f!y::FY, jac, backend::AbstractADType, x, extras::PullbackJacobianExtras{B}
) where {FY,B}
    @compat (; batched_seeds, batched_results, pullback_extras, M) = extras

    pullback_extras_same = prepare_pullback_same_point(
        f_or_f!y..., backend, x, batched_seeds[1], extras.pullback_extras
    )

    for a in eachindex(batched_seeds, batched_results)
        pullback!(
            f_or_f!y...,
            batched_results[a],
            backend,
            x,
            batched_seeds[a],
            pullback_extras_same,
        )

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(jac, 1 + ((a - 1) * B + (b - 1)) % M, :), vec(batched_results[a].d[b])
            )
        end
    end

    return jac
end
