## Docstrings

"""
    prepare_jacobian(f,     backend, x) -> extras
    prepare_jacobian(f!, y, backend, x) -> extras

Create an `extras` object that can be given to [`jacobian`](@ref) and its variants.

!!! warning
    If the function changes in any way, the result of preparation will be invalidated, and you will need to run it again.
    For in-place functions, `y` is mutated by `f!` during preparation.
"""
function prepare_jacobian end

"""
    value_and_jacobian(f,     [extras,] backend, x) -> (y, jac)
    value_and_jacobian(f!, y, [extras,] backend, x) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`.

$(document_preparation("jacobian"))
"""
function value_and_jacobian end

"""
    value_and_jacobian!(f,     jac, [extras,] backend, x) -> (y, jac)
    value_and_jacobian!(f!, y, jac, [extras,] backend, x) -> (y, jac)

Compute the value and the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.
    
$(document_preparation("jacobian"))
"""
function value_and_jacobian! end

"""
    jacobian(f,     [extras,] backend, x) -> jac
    jacobian(f!, y, [extras,] backend, x) -> jac

Compute the Jacobian matrix of the function `f` at point `x`.

$(document_preparation("jacobian"))
"""
function jacobian end

"""
    jacobian!(f,     jac, [extras,] backend, x) -> jac
    jacobian!(f!, y, jac, [extras,] backend, x) -> jac

Compute the Jacobian matrix of the function `f` at point `x`, overwriting `jac`.

$(document_preparation("jacobian"))
"""
function jacobian! end

## Preparation

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

function prepare_jacobian(
    f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    y = f(x, map(unwrap, contexts)...)
    return _prepare_jacobian_aux(
        pushforward_performance(backend), y, (f,), backend, x, contexts...
    )
end

function prepare_jacobian(
    f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return _prepare_jacobian_aux(
        pushforward_performance(backend), y, (f!, y), backend, x, contexts...
    )
end

function _prepare_jacobian_aux(
    ::PushforwardFast,
    y,
    f_or_f!y::FY,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,C}
    N = length(x)
    B = pick_batchsize(backend, N)
    seeds = [basis(backend, x, ind) for ind in eachindex(x)]
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B))...) for
        a in 1:div(N, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(y), Val(B))...) for _ in batched_seeds]
    pushforward_extras = prepare_pushforward(
        f_or_f!y..., backend, x, batched_seeds[1], contexts...
    )
    D = eltype(batched_seeds[1])
    R = eltype(batched_results[1])
    E = typeof(pushforward_extras)
    return PushforwardJacobianExtras{B,D,R,E}(
        batched_seeds, batched_results, pushforward_extras, N
    )
end

function _prepare_jacobian_aux(
    ::PushforwardSlow,
    y,
    f_or_f!y::FY,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,C}
    M = length(y)
    B = pick_batchsize(backend, M)
    seeds = [basis(backend, y, ind) for ind in eachindex(y)]
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % M], Val(B))...) for
        a in 1:div(M, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(x), Val(B))...) for _ in batched_seeds]
    pullback_extras = prepare_pullback(
        f_or_f!y..., backend, x, batched_seeds[1], contexts...
    )
    D = eltype(batched_seeds[1])
    R = eltype(batched_results[1])
    E = typeof(pullback_extras)
    return PullbackJacobianExtras{B,D,R,E}(
        batched_seeds, batched_results, pullback_extras, M
    )
end

## One argument

function jacobian(
    f::F, extras::JacobianExtras, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return _jacobian_aux((f,), extras, backend, x)
end

function jacobian!(
    f::F,
    jac,
    extras::JacobianExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return _jacobian_aux!((f,), jac, extras, backend, x, contexts...)
end

function value_and_jacobian(
    f::F, extras::JacobianExtras, backend::AbstractADType, x, contexts::Vararg{Context,C}
) where {F,C}
    return f(x), jacobian(f, extras, backend, x, contexts...)
end

function value_and_jacobian!(
    f::F,
    jac,
    extras::JacobianExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return f(x, map(unwrap, contexts)...),
    jacobian!(f, jac, extras, backend, x, contexts...)
end

## Two arguments

function jacobian(
    f!::F,
    y,
    extras::JacobianExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return _jacobian_aux((f!, y), extras, backend, x, contexts...)
end

function jacobian!(
    f!::F,
    y,
    jac,
    extras::JacobianExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return _jacobian_aux!((f!, y), jac, extras, backend, x, contexts...)
end

function value_and_jacobian(
    f!::F,
    y,
    extras::JacobianExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    jac = jacobian(f!, y, extras, backend, x, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    return y, jac
end

function value_and_jacobian!(
    f!::F,
    y,
    jac,
    extras::JacobianExtras,
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    jacobian!(f!, y, jac, extras, backend, x, contexts...)
    f!(y, x, map(unwrap, contexts)...)
    return y, jac
end

## Common auxiliaries

function _jacobian_aux(
    f_or_f!y::FY,
    extras::PushforwardJacobianExtras{B},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    @compat (; batched_seeds, pushforward_extras, N) = extras

    pushforward_extras_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_extras, backend, x, batched_seeds[1], contexts...
    )

    jac_blocks = map(eachindex(batched_seeds)) do a
        dy_batch = pushforward(
            f_or_f!y...,
            pushforward_extras_same,
            backend,
            x,
            batched_seeds[a],
            contexts...,
        )
        stack(vec, dy_batch.d; dims=2)
    end

    jac = reduce(hcat, jac_blocks)
    if N < size(jac, 2)
        jac = jac[:, 1:N]
    end
    return jac
end

function _jacobian_aux(
    f_or_f!y::FY,
    extras::PullbackJacobianExtras{B},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    @compat (; batched_seeds, pullback_extras, M) = extras

    pullback_extras_same = prepare_pullback_same_point(
        f_or_f!y..., extras.pullback_extras, backend, x, batched_seeds[1], contexts...
    )

    jac_blocks = map(eachindex(batched_seeds)) do a
        dx_batch = pullback(
            f_or_f!y..., pullback_extras_same, backend, x, batched_seeds[a], contexts...
        )
        stack(vec, dx_batch.d; dims=1)
    end

    jac = reduce(vcat, jac_blocks)
    if M < size(jac, 1)
        jac = jac[1:M, :]
    end
    return jac
end

function _jacobian_aux!(
    f_or_f!y::FY,
    jac,
    extras::PushforwardJacobianExtras{B},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    @compat (; batched_seeds, batched_results, pushforward_extras, N) = extras

    pushforward_extras_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_extras, backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        pushforward!(
            f_or_f!y...,
            batched_results[a],
            pushforward_extras_same,
            backend,
            x,
            batched_seeds[a],
            contexts...,
        )

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(jac, :, 1 + ((a - 1) * B + (b - 1)) % N), vec(batched_results[a].d[b])
            )
        end
    end

    return jac
end

function _jacobian_aux!(
    f_or_f!y::FY,
    jac,
    extras::PullbackJacobianExtras{B},
    backend::AbstractADType,
    x,
    contexts::Vararg{Context,C},
) where {FY,B,C}
    @compat (; batched_seeds, batched_results, pullback_extras, M) = extras

    pullback_extras_same = prepare_pullback_same_point(
        f_or_f!y..., extras.pullback_extras, backend, x, batched_seeds[1], contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        pullback!(
            f_or_f!y...,
            batched_results[a],
            pullback_extras_same,
            backend,
            x,
            batched_seeds[a],
            contexts...,
        )

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(jac, 1 + ((a - 1) * B + (b - 1)) % M, :), vec(batched_results[a].d[b])
            )
        end
    end

    return jac
end
