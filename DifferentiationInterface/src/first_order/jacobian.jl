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

function prepare_jacobian(f::F, backend::AbstractADType, x) where {F}
    y = f(x)
    return _prepare_jacobian_aux((f,), backend, x, y, pushforward_performance(backend))
end

function prepare_jacobian(f!::F, y, backend::AbstractADType, x) where {F}
    return _prepare_jacobian_aux((f!, y), backend, x, y, pushforward_performance(backend))
end

function _prepare_jacobian_aux(
    f_or_f!y::FY, backend::AbstractADType, x, y, ::PushforwardFast
) where {FY}
    N = length(x)
    B = pick_batchsize(backend, N)
    seeds = [basis(backend, x, ind) for ind in CartesianIndices(x)]
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B))) for
        a in 1:div(N, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(y), Val(B))) for _ in batched_seeds]
    pushforward_extras = prepare_pushforward(f_or_f!y..., backend, x, batched_seeds[1])
    D = tuptype(batched_seeds[1])
    R = tuptype(batched_results[1])
    E = typeof(pushforward_extras)
    return PushforwardJacobianExtras{B,D,R,E}(
        batched_seeds, batched_results, pushforward_extras, N
    )
end

function _prepare_jacobian_aux(
    f_or_f!y::FY, backend::AbstractADType, x, y, ::PushforwardSlow
) where {FY}
    M = length(y)
    B = pick_batchsize(backend, M)
    seeds = [basis(backend, y, ind) for ind in CartesianIndices(y)]
    batched_seeds = [
        Tangents(ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % M], Val(B))) for
        a in 1:div(M, B, RoundUp)
    ]
    batched_results = [Tangents(ntuple(b -> similar(x), Val(B))) for _ in batched_seeds]
    pullback_extras = prepare_pullback(f_or_f!y..., backend, x, batched_seeds[1])
    D = tuptype(batched_seeds[1])
    R = tuptype(batched_results[1])
    E = typeof(pullback_extras)
    return PullbackJacobianExtras{B,D,R,E}(
        batched_seeds, batched_results, pullback_extras, M
    )
end

## One argument

function jacobian(f::F, extras::JacobianExtras, backend::AbstractADType, x) where {F}
    return _jacobian_aux((f,), extras, backend, x)
end

function jacobian!(f::F, jac, extras::JacobianExtras, backend::AbstractADType, x) where {F}
    return _jacobian_aux!((f,), jac, extras, backend, x)
end

function value_and_jacobian(
    f::F, extras::JacobianExtras, backend::AbstractADType, x
) where {F}
    return f(x), jacobian(f, extras, backend, x)
end

function value_and_jacobian!(
    f::F, jac, extras::JacobianExtras, backend::AbstractADType, x
) where {F}
    return f(x), jacobian!(f, jac, extras, backend, x)
end

## Two arguments

function jacobian(f!::F, y, extras::JacobianExtras, backend::AbstractADType, x) where {F}
    return _jacobian_aux((f!, y), extras, backend, x)
end

function jacobian!(
    f!::F, y, jac, extras::JacobianExtras, backend::AbstractADType, x
) where {F}
    return _jacobian_aux!((f!, y), jac, extras, backend, x)
end

function value_and_jacobian(
    f!::F, y, extras::JacobianExtras, backend::AbstractADType, x
) where {F}
    jac = jacobian(f!, y, extras, backend, x)
    f!(y, x)
    return y, jac
end

function value_and_jacobian!(
    f!::F, y, jac, extras::JacobianExtras, backend::AbstractADType, x
) where {F}
    jacobian!(f!, y, jac, extras, backend, x)
    f!(y, x)
    return y, jac
end

## Common auxiliaries

function _jacobian_aux(
    f_or_f!y::FY, extras::PushforwardJacobianExtras{B}, backend::AbstractADType, x
) where {FY,B}
    @compat (; batched_seeds, pushforward_extras, N) = extras

    pushforward_extras_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_extras, backend, x, batched_seeds[1]
    )

    jac_blocks = map(eachindex(batched_seeds)) do a
        dy_batch = pushforward(
            f_or_f!y..., pushforward_extras_same, backend, x, batched_seeds[a]
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
    f_or_f!y::FY, extras::PullbackJacobianExtras{B}, backend::AbstractADType, x
) where {FY,B}
    @compat (; batched_seeds, pullback_extras, M) = extras

    pullback_extras_same = prepare_pullback_same_point(
        f_or_f!y..., extras.pullback_extras, backend, x, batched_seeds[1]
    )

    jac_blocks = map(eachindex(batched_seeds)) do a
        dx_batch = pullback(f_or_f!y..., pullback_extras_same, backend, x, batched_seeds[a])
        stack(vec, dx_batch.d; dims=1)
    end

    jac = reduce(vcat, jac_blocks)
    if M < size(jac, 1)
        jac = jac[1:M, :]
    end
    return jac
end

function _jacobian_aux!(
    f_or_f!y::FY, jac, extras::PushforwardJacobianExtras{B}, backend::AbstractADType, x
) where {FY,B}
    @compat (; batched_seeds, batched_results, pushforward_extras, N) = extras

    pushforward_extras_same = prepare_pushforward_same_point(
        f_or_f!y..., pushforward_extras, backend, x, batched_seeds[1]
    )

    for a in eachindex(batched_seeds, batched_results)
        pushforward!(
            f_or_f!y...,
            batched_results[a],
            pushforward_extras_same,
            backend,
            x,
            batched_seeds[a],
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
    f_or_f!y::FY, jac, backend::AbstractADType, x, extras::PullbackJacobianExtras{B}
) where {FY,B}
    @compat (; batched_seeds, batched_results, pullback_extras, M) = extras

    pullback_extras_same = prepare_pullback_same_point(
        f_or_f!y..., extras.pullback_extras, backend, x, batched_seeds[1]
    )

    for a in eachindex(batched_seeds, batched_results)
        pullback!(
            f_or_f!y...,
            batched_results[a],
            pullback_extras_same,
            backend,
            x,
            batched_seeds[a],
        )

        for b in eachindex(batched_results[a].d)
            copyto!(
                view(jac, 1 + ((a - 1) * B + (b - 1)) % M, :), vec(batched_results[a].d[b])
            )
        end
    end

    return jac
end
