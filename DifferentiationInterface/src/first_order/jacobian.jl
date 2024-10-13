## Docstrings

"""
    prepare_jacobian(f,     backend, x, [contexts...]) -> prep
    prepare_jacobian(f!, y, backend, x, [contexts...]) -> prep

Create a `prep` object that can be given to [`jacobian`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_jacobian end

"""
    prepare!_jacobian(f,     prep, backend, x, [contexts...]) -> new_prep
    prepare!_jacobian(f!, y, prep, backend, x, [contexts...]) -> new_prep

Same behavior as [`prepare_jacobian`](@ref) but can modify an existing `prep` object to avoid some allocations.

There is no guarantee that `prep` will be mutated, or that performance will be improved compared to preparation from scratch.

!!! danger
    For efficiency, this function needs to rely on backend package internals, therefore it not protected by semantic versioning.
"""
function prepare!_jacobian end

"""
    value_and_jacobian(f,     [prep,] backend, x, [contexts...]) -> (y, jac)
    value_and_jacobian(f!, y, [prep,] backend, x, [contexts...]) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`.

$(document_preparation("jacobian"))
"""
function value_and_jacobian end

"""
    value_and_jacobian!(f,     jac, [prep,] backend, x, [contexts...]) -> (y, jac)
    value_and_jacobian!(f!, y, jac, [prep,] backend, x, [contexts...]) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.
    
$(document_preparation("jacobian"))
"""
function value_and_jacobian! end

"""
    jacobian(f,     [prep,] backend, x, [contexts...]) -> jac
    jacobian(f!, y, [prep,] backend, x, [contexts...]) -> jac

Compute the Jacobian matrix of the function `f` at point `x`.

$(document_preparation("jacobian"))
"""
function jacobian end

"""
    jacobian!(f,     jac, [prep,] backend, x, [contexts...]) -> jac
    jacobian!(f!, y, jac, [prep,] backend, x, [contexts...]) -> jac

Compute the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.

$(document_preparation("jacobian"))
"""
function jacobian! end

## Preparation

struct PushforwardJacobianPrep{B,TD<:NTuple{B},TR<:NTuple{B},E<:PushforwardPrep} <:
       JacobianPrep
    batched_seeds::Vector{TD}
    batched_results::Vector{TR}
    pushforward_prep::E
    N::Int
end

struct PullbackJacobianPrep{B,TD<:NTuple{B},TR<:NTuple{B},E<:PullbackPrep} <: JacobianPrep
    batched_seeds::Vector{TD}
    batched_results::Vector{TR}
    pullback_prep::E
    M::Int
end

function prepare_jacobian(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    y = f(x, map(unwrap, contexts)...)
    perf = pushforward_performance(backend)
    # type-unstable
    if perf isa PushforwardFast
        valB = pick_batchsize(backend, x)
    else
        valB = pick_batchsize(backend, y)
    end
    # function barrier
    return _prepare_jacobian_aux(perf, valB, y, (f,), backend, x, contexts...)
end

function prepare_jacobian(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    perf = pushforward_performance(backend)
    # type-unstable
    if perf isa PushforwardFast
        valB = pick_batchsize(backend, x)
    else
        valB = pick_batchsize(backend, y)
    end
    # function barrier
    return _prepare_jacobian_aux(perf, valB, y, (f!, y), backend, x, contexts...)
end

function _prepare_jacobian_aux(
    ::PushforwardFast,
    ::Val{B},
    y,
    f_or_f!y::FY,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {B,FY,C}
    N = length(x)
    seeds = [basis(backend, x, ind) for ind in eachindex(x)]
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for
        a in 1:div(N, B, RoundUp)
    ]
    batched_results = [ntuple(b -> similar(y), Val(B)) for _ in batched_seeds]
    pushforward_prep = prepare_pushforward(
        f_or_f!y..., backend, x, batched_seeds[1], contexts...
    )
    TD = eltype(batched_seeds)
    TR = eltype(batched_results)
    E = typeof(pushforward_prep)
    return PushforwardJacobianPrep{B,TD,TR,E}(
        batched_seeds, batched_results, pushforward_prep, N
    )
end

function _prepare_jacobian_aux(
    ::PushforwardSlow,
    ::Val{B},
    y,
    f_or_f!y::FY,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {B,FY,C}
    M = length(y)
    seeds = [basis(backend, y, ind) for ind in eachindex(y)]
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % M], Val(B)) for
        a in 1:div(M, B, RoundUp)
    ]
    batched_results = [ntuple(b -> similar(x), Val(B)) for _ in batched_seeds]
    pullback_prep = prepare_pullback(f_or_f!y..., backend, x, batched_seeds[1], contexts...)
    TD = eltype(batched_seeds)
    TR = eltype(batched_results)
    E = typeof(pullback_prep)
    return PullbackJacobianPrep{B,TD,TR,E}(batched_seeds, batched_results, pullback_prep, M)
end

## One argument

function jacobian(
    f::F, prep::JacobianPrep, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return _jacobian_aux((f,), prep, backend, x, contexts...)
end

function jacobian!(
    f::F, jac, prep::JacobianPrep, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return _jacobian_aux!((f,), jac, prep, backend, x, contexts...)
end

function value_and_jacobian(
    f::F, prep::JacobianPrep, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return f(x, map(unwrap, contexts)...), jacobian(f, prep, backend, x, contexts...)
end

function value_and_jacobian!(
    f::F, jac, prep::JacobianPrep, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return f(x, map(unwrap, contexts)...), jacobian!(f, jac, prep, backend, x, contexts...)
end

## Two arguments

function jacobian(
    f!::F, y, prep::JacobianPrep, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return _jacobian_aux((f!, y), prep, backend, x, contexts...)
end

function jacobian!(
    f!::F,
    y,
    jac,
    prep::JacobianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return _jacobian_aux!((f!, y), jac, prep, backend, x, contexts...)
end

function value_and_jacobian(
    f!::F, y, prep::JacobianPrep, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    jac = jacobian(f!, y, prep, backend, x, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    return y, jac
end

function value_and_jacobian!(
    f!::F,
    y,
    jac,
    prep::JacobianPrep,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    jacobian!(f!, y, jac, prep, backend, x, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    return y, jac
end

## Common auxiliaries

function _jacobian_aux(
    f_or_f!y::FY,
    prep::PushforwardJacobianPrep{B},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    (; batched_seeds, pushforward_prep, N) = prep

    pushforward_prep_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_prep, backend, x, batched_seeds[1], contexts...
    )

    jac = mapreduce(hcat, eachindex(batched_seeds)) do a
        dy_batch = pushforward(
            f_or_f!y...,
            pushforward_prep_same,
            backend,
            x,
            batched_seeds[a],
            contexts...,
        )
        block = stack_vec_col(dy_batch)
        if N % B != 0 && a == lastindex(batched_seeds)
            block = block[:, 1:(N - (a - 1) * B)]
        end
        block
    end
    return jac
end

function _jacobian_aux(
    f_or_f!y::FY,
    prep::PullbackJacobianPrep{B},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    (; batched_seeds, pullback_prep, M) = prep

    pullback_prep_same = prepare_pullback_same_point(
        f_or_f!y..., prep.pullback_prep, backend, x, batched_seeds[1], contexts...
    )

    jac = mapreduce(vcat, eachindex(batched_seeds)) do a
        dx_batch = pullback(
            f_or_f!y..., pullback_prep_same, backend, x, batched_seeds[a], contexts...
        )
        block = stack_vec_row(dx_batch)
        if M % B != 0 && a == lastindex(batched_seeds)
            block = block[1:(M - (a - 1) * B), :]
        end
        block
    end
    return jac
end

function _jacobian_aux!(
    f_or_f!y::FY,
    jac,
    prep::PushforwardJacobianPrep{B},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    (; batched_seeds, batched_results, pushforward_prep, N) = prep

    pushforward_prep_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_prep, backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        pushforward!(
            f_or_f!y...,
            batched_results[a],
            pushforward_prep_same,
            backend,
            x,
            batched_seeds[a],
            contexts...,
        )

        for b in eachindex(batched_results[a])
            copyto!(
                view(jac, :, 1 + ((a - 1) * B + (b - 1)) % N), vec(batched_results[a][b])
            )
        end
    end

    return jac
end

function _jacobian_aux!(
    f_or_f!y::FY,
    jac,
    prep::PullbackJacobianPrep{B},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    (; batched_seeds, batched_results, pullback_prep, M) = prep

    pullback_prep_same = prepare_pullback_same_point(
        f_or_f!y..., prep.pullback_prep, backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        pullback!(
            f_or_f!y...,
            batched_results[a],
            pullback_prep_same,
            backend,
            x,
            batched_seeds[a],
            contexts...,
        )

        for b in eachindex(batched_results[a])
            copyto!(
                view(jac, 1 + ((a - 1) * B + (b - 1)) % M, :), vec(batched_results[a][b])
            )
        end
    end

    return jac
end
