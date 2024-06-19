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

struct PushforwardJacobianExtras{B,E<:PushforwardExtras} <: JacobianExtras
    pushforward_batched_extras::E
end

struct PullbackJacobianExtras{B,E<:PullbackExtras} <: JacobianExtras
    pullback_batched_extras::E
end

function prepare_jacobian(f::F, backend::AbstractADType, x) where {F}
    return prepare_jacobian_aux(f, backend, x, pushforward_performance(backend))
end

function prepare_jacobian(f!::F, y, backend::AbstractADType, x) where {F}
    return prepare_jacobian_aux(f!, y, backend, x, pushforward_performance(backend))
end

function prepare_jacobian_aux(f::F, backend, x, ::PushforwardFast) where {F}
    N = length(x)
    B = pick_batchsize(backend, N)
    dx = basis(backend, x, first(CartesianIndices(x)))
    dx_batch = Batch(ntuple(Returns(dx), Val{B}()))
    pushforward_batched_extras = prepare_pushforward_batched(f, backend, x, dx_batch)
    E = typeof(pushforward_batched_extras)
    return PushforwardJacobianExtras{B,E}(pushforward_batched_extras)
end

function prepare_jacobian_aux(f::F, backend, x, ::PushforwardSlow) where {F}
    y = f(x)
    M = length(y)
    B = pick_batchsize(backend, M)
    dy = basis(backend, y, first(CartesianIndices(y)))
    dy_batch = Batch(ntuple(Returns(dy), Val{B}()))
    pullback_batched_extras = prepare_pullback_batched(f, backend, x, dy_batch)
    E = typeof(pullback_batched_extras)
    return PullbackJacobianExtras{B,E}(pullback_batched_extras)
end

function prepare_jacobian_aux(f!::F, y, backend, x, ::PushforwardFast) where {F}
    N = length(x)
    B = pick_batchsize(backend, N)
    dx = basis(backend, x, first(CartesianIndices(x)))
    dx_batch = Batch(ntuple(Returns(dx), Val{B}()))
    pushforward_batched_extras = prepare_pushforward_batched(f!, y, backend, x, dx_batch)
    E = typeof(pushforward_batched_extras)
    return PushforwardJacobianExtras{B,E}(pushforward_batched_extras)
end

function prepare_jacobian_aux(f!::F, y, backend, x, ::PushforwardSlow) where {F}
    M = length(y)
    B = pick_batchsize(backend, M)
    dy = basis(backend, y, first(CartesianIndices(y)))
    dy_batch = Batch(ntuple(Returns(dy), Val{B}()))
    pullback_batched_extras = prepare_pullback_batched(f!, y, backend, x, dy_batch)
    E = typeof(pullback_batched_extras)
    return PullbackJacobianExtras{B,E}(pullback_batched_extras)
end

## One argument

function value_and_jacobian(f::F, backend::AbstractADType, x) where {F}
    return value_and_jacobian(f, backend, x, prepare_jacobian(f, backend, x))
end

function value_and_jacobian!(f::F, jac, backend::AbstractADType, x) where {F}
    return value_and_jacobian!(f, jac, backend, x, prepare_jacobian(f, backend, x))
end

function jacobian(f::F, backend::AbstractADType, x) where {F}
    return jacobian(f, backend, x, prepare_jacobian(f, backend, x))
end

function jacobian!(f::F, jac, backend::AbstractADType, x) where {F}
    return jacobian!(f, jac, backend, x, prepare_jacobian(f, backend, x))
end

function value_and_jacobian(
    f::F, backend, x::AbstractArray, extras::PushforwardJacobianExtras{B}
) where {F,B}
    y = f(x)  # TODO: remove
    xinds = CartesianIndices(x)
    N = length(x)

    example_dx_batch_elements = ntuple(Returns(basis(backend, x, xinds[1])), Val{B}())
    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f, backend, x, Batch(example_dx_batch_elements), extras.pushforward_batched_extras
    )

    jac = mapreduce(hcat, 1:div(N, B, RoundUp)) do a
        dx_batch_elements = ntuple(Val{B}()) do b
            basis(backend, x, xinds[1 + ((a - 1) * B + (b - 1)) % N])
        end
        dy_batch = pushforward_batched(
            f, backend, x, Batch(dx_batch_elements), pushforward_batched_extras_same
        )
        stack(vec, dy_batch.elements; dims=2)
    end

    if N < size(jac, 2)
        jac = jac[:, 1:N]
    end
    return y, jac
end

function value_and_jacobian(
    f::F, backend, x::AbstractArray, extras::PullbackJacobianExtras{B}
) where {F,B}
    y = f(x)  # TODO: remove
    yinds = CartesianIndices(y)
    M = length(y)

    example_dy_batch_elements = ntuple(Returns(basis(backend, y, yinds[1])), Val{B}())
    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f, backend, x, Batch(example_dy_batch_elements), extras.pullback_batched_extras
    )

    jac = mapreduce(vcat, 1:div(M, B, RoundUp)) do a
        dy_batch_elements = ntuple(Val{B}()) do b
            basis(backend, y, yinds[1 + ((a - 1) * B + (b - 1)) % M])
        end
        dx_batch = pullback_batched(
            f, backend, x, Batch(dy_batch_elements), pullback_batched_extras_same
        )
        stack(vec, dx_batch.elements; dims=1)
    end

    if M < size(jac, 1)
        jac = jac[1:M, :]
    end
    return y, jac
end

function value_and_jacobian!(
    f::F,
    jac::AbstractMatrix,
    backend,
    x::AbstractArray,
    extras::PushforwardJacobianExtras{B},
) where {F,B}
    y = f(x)  # TODO: remove
    xinds = CartesianIndices(x)
    N = length(x)

    example_dx_batch_elements = ntuple(Returns(basis(backend, x, xinds[1])), Val{B}())
    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f, backend, x, Batch(example_dx_batch_elements), extras.pushforward_batched_extras
    )

    for a in 1:div(N, B, RoundUp)
        dx_batch_elements = ntuple(Val{B}()) do b
            basis(backend, x, xinds[1 + ((a - 1) * B + (b - 1)) % N])
        end
        dy_batch_elements = ntuple(Val{B}()) do b
            reshape(view(jac, :, 1 + ((a - 1) * B + (b - 1)) % N), size(y))
        end
        pushforward_batched!(
            f,
            Batch(dy_batch_elements),
            backend,
            x,
            Batch(dx_batch_elements),
            pushforward_batched_extras_same,
        )
    end

    return y, jac
end

function value_and_jacobian!(
    f::F, jac::AbstractMatrix, backend, x::AbstractArray, extras::PullbackJacobianExtras{B}
) where {F,B}
    y = f(x)  # TODO: remove
    yinds = CartesianIndices(y)
    M = length(y)

    example_dy_batch_elements = ntuple(Returns(basis(backend, y, yinds[1])), Val{B}())
    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f, backend, x, Batch(example_dy_batch_elements), extras.pullback_batched_extras
    )

    for a in 1:div(M, B, RoundUp)
        dy_batch_elements = ntuple(Val{B}()) do b
            basis(backend, y, yinds[1 + ((a - 1) * B + (b - 1)) % M])
        end
        dx_batch_elements = ntuple(Val{B}()) do b
            reshape(view(jac, 1 + ((a - 1) * B + (b - 1)) % M, :), size(x))
        end
        pullback_batched!(
            f,
            Batch(dx_batch_elements),
            backend,
            x,
            Batch(dy_batch_elements),
            pullback_batched_extras_same,
        )
    end

    return y, jac
end

function jacobian(f::F, backend::AbstractADType, x, extras::JacobianExtras) where {F}
    return value_and_jacobian(f, backend, x, extras)[2]
end

function jacobian!(f::F, jac, backend::AbstractADType, x, extras::JacobianExtras) where {F}
    return value_and_jacobian!(f, jac, backend, x, extras)[2]
end

## Two arguments

function value_and_jacobian(f!::F, y, backend::AbstractADType, x) where {F}
    return value_and_jacobian(f!, y, backend, x, prepare_jacobian(f!, y, backend, x))
end

function value_and_jacobian!(f!::F, y, jac, backend::AbstractADType, x) where {F}
    return value_and_jacobian!(f!, y, jac, backend, x, prepare_jacobian(f!, y, backend, x))
end

function jacobian(f!::F, y, backend::AbstractADType, x) where {F}
    return jacobian(f!, y, backend, x, prepare_jacobian(f!, y, backend, x))
end

function jacobian!(f!::F, y, jac, backend::AbstractADType, x) where {F}
    return jacobian!(f!, y, jac, backend, x, prepare_jacobian(f!, y, backend, x))
end

function jacobian(
    f!::F, y, backend, x::AbstractArray, extras::PushforwardJacobianExtras{B}
) where {F,B}
    xinds = CartesianIndices(x)
    N = length(x)

    example_dx_batch_elements = ntuple(Returns(basis(backend, x, xinds[1])), Val{B}())
    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f!,
        y,
        backend,
        x,
        Batch(example_dx_batch_elements),
        extras.pushforward_batched_extras,
    )

    jac = mapreduce(hcat, 1:div(N, B, RoundUp)) do a
        dx_batch_elements = ntuple(Val{B}()) do b
            basis(backend, x, xinds[1 + ((a - 1) * B + (b - 1)) % N])
        end
        dy_batch = pushforward_batched(
            f!, y, backend, x, Batch(dx_batch_elements), pushforward_batched_extras_same
        )
        stack(vec, dy_batch.elements; dims=2)
    end

    if N < size(jac, 2)
        jac = jac[:, 1:N]
    end
    return jac
end

function jacobian(
    f!::F, y, backend, x::AbstractArray, extras::PullbackJacobianExtras{B}
) where {F,B}
    yinds = CartesianIndices(y)
    M = length(y)

    example_dy_batch_elements = ntuple(Returns(basis(backend, y, yinds[1])), Val{B}())
    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f!, y, backend, x, Batch(example_dy_batch_elements), extras.pullback_batched_extras
    )

    jac = mapreduce(vcat, 1:div(M, B, RoundUp)) do a
        dy_batch_elements = ntuple(Val{B}()) do b
            basis(backend, y, yinds[1 + ((a - 1) * B + (b - 1)) % M])
        end
        dx_batch = pullback_batched(
            f!, y, backend, x, Batch(dy_batch_elements), pullback_batched_extras_same
        )
        stack(vec, dx_batch.elements; dims=1)
    end

    if M < size(jac, 1)
        jac = jac[1:M, :]
    end
    return jac
end

function jacobian!(
    f!::F,
    y,
    jac::AbstractMatrix,
    backend,
    x::AbstractArray,
    extras::PushforwardJacobianExtras{B},
) where {F,B}
    xinds = CartesianIndices(x)
    N = length(x)

    example_dx_batch_elements = ntuple(Returns(basis(backend, x, xinds[1])), Val{B}())
    pushforward_batched_extras_same = prepare_pushforward_batched_same_point(
        f!,
        y,
        backend,
        x,
        Batch(example_dx_batch_elements),
        extras.pushforward_batched_extras,
    )

    for a in 1:div(N, B, RoundUp)
        dx_batch_elements = ntuple(Val{B}()) do b
            basis(backend, x, xinds[1 + ((a - 1) * B + (b - 1)) % N])
        end
        dy_batch_elements = ntuple(Val{B}()) do b
            reshape(view(jac, :, 1 + ((a - 1) * B + (b - 1)) % N), size(y))
        end
        pushforward_batched!(
            f!,
            y,
            Batch(dy_batch_elements),
            backend,
            x,
            Batch(dx_batch_elements),
            pushforward_batched_extras_same,
        )
    end

    return jac
end

function jacobian!(
    f!::F,
    y,
    jac::AbstractMatrix,
    backend,
    x::AbstractArray,
    extras::PullbackJacobianExtras{B},
) where {F,B}
    yinds = CartesianIndices(y)
    M = length(y)

    example_dy_batch_elements = Batch(
        ntuple(Returns(basis(backend, y, yinds[1])), Val{B}())
    )
    pullback_batched_extras_same = prepare_pullback_batched_same_point(
        f!, y, backend, x, Batch(example_dy_batch_elements), extras.pullback_batched_extras
    )

    for a in 1:div(M, B, RoundUp)
        dy_batch_elements = ntuple(Val{B}()) do b
            basis(backend, y, yinds[1 + ((a - 1) * B + (b - 1)) % M])
        end
        dx_batch_elements = ntuple(Val{B}()) do b
            reshape(view(jac, 1 + ((a - 1) * B + (b - 1)) % M, :), size(x))
        end
        pullback_batched!(
            f!,
            y,
            Batch(dx_batch_elements),
            backend,
            x,
            Batch(dy_batch_elements),
            pullback_batched_extras_same,
        )
    end

    return jac
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
