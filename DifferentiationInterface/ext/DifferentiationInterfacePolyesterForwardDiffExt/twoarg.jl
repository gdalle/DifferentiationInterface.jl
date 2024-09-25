## Pushforward

function DI.prepare_pushforward(
    f!, y, backend::AutoPolyesterForwardDiff, x, tx::Tangents, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_pushforward(f!, y, single_threaded(backend), x, tx, contexts...)
end

function DI.value_and_pushforward(
    f!,
    y,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_pushforward(
        f!, y, prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.value_and_pushforward!(
    f!,
    y,
    ty::Tangents,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_pushforward!(
        f!, y, ty, prep, single_threaded(backend), x, tx, contexts...
    )
end

function DI.pushforward(
    f!,
    y,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {C}
    return DI.pushforward(f!, y, prep, single_threaded(backend), x, tx, contexts...)
end

function DI.pushforward!(
    f!,
    y,
    ty::Tangents,
    prep::PushforwardPrep,
    backend::AutoPolyesterForwardDiff,
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {C}
    return DI.pushforward!(f!, y, ty, prep, single_threaded(backend), x, tx, contexts...)
end

## Derivative

function DI.prepare_derivative(
    f!, y, backend::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return DI.prepare_derivative(f!, y, single_threaded(backend), x, contexts...)
end

function DI.value_and_derivative(
    f!,
    y,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_derivative(f!, y, prep, single_threaded(backend), x, contexts...)
end

function DI.value_and_derivative!(
    f!,
    y,
    der,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.value_and_derivative!(
        f!, y, der, prep, single_threaded(backend), x, contexts...
    )
end

function DI.derivative(
    f!,
    y,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.derivative(f!, y, prep, single_threaded(backend), x, contexts...)
end

function DI.derivative!(
    f!,
    y,
    der,
    prep::DerivativePrep,
    backend::AutoPolyesterForwardDiff,
    x,
    contexts::Vararg{Context,C},
) where {C}
    return DI.derivative!(f!, y, der, prep, single_threaded(backend), x, contexts...)
end

## Jacobian

function DI.prepare_jacobian(
    f!, y, ::AutoPolyesterForwardDiff, x, contexts::Vararg{Context,C}
) where {C}
    return NoJacobianPrep()
end

function DI.value_and_jacobian(
    f!, y, ::NoJacobianPrep, ::AutoPolyesterForwardDiff{K}, x, contexts::Vararg{Context,C}
) where {K,C}
    fc! = with_contexts(f!, contexts...)
    jac = similar(y, length(y), length(x))
    threaded_jacobian!(fc!, y, jac, x, Chunk{K}())
    fc!(y, x)
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    ::NoJacobianPrep,
    ::AutoPolyesterForwardDiff{K},
    x,
    contexts::Vararg{Context,C},
) where {K,C}
    fc! = with_contexts(f!, contexts...)
    threaded_jacobian!(fc!, y, jac, x, Chunk{K}())
    fc!(y, x)
    return y, jac
end

function DI.jacobian(
    f!, y, ::NoJacobianPrep, ::AutoPolyesterForwardDiff{K}, x, contexts::Vararg{Context,C}
) where {K,C}
    fc! = with_contexts(f!, contexts...)
    jac = similar(y, length(y), length(x))
    threaded_jacobian!(fc!, y, jac, x, Chunk{K}())
    return jac
end

function DI.jacobian!(
    f!,
    y,
    jac,
    ::NoJacobianPrep,
    ::AutoPolyesterForwardDiff{K},
    x,
    contexts::Vararg{Context,C},
) where {K,C}
    fc! = with_contexts(f!, contexts...)
    threaded_jacobian!(fc!, y, jac, x, Chunk{K}())
    return jac
end
