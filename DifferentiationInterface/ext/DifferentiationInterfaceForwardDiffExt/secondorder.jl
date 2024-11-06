struct ForwardDiffOverSomethingHVPPrep{E1<:GradientPrep,E2<:PushforwardPrep} <: HVPPrep
    inner_gradient_prep::E1
    outer_pushforward_prep::E2
end

function DI.prepare_hvp(
    f::F,
    backend::SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    T = tag_type(shuffled_gradient, outer(backend), x)
    xdual = make_dual(T, x, tx)
    inner_gradient_prep = DI.prepare_gradient(f, inner(backend), xdual, contexts...)
    rewrap = Rewrap(contexts...)
    new_contexts = (
        Constant(f),
        PrepContext(inner_gradient_prep),
        Constant(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    outer_pushforward_prep = DI.prepare_pushforward(
        shuffled_gradient, outer(backend), x, tx, new_contexts...
    )
    return ForwardDiffOverSomethingHVPPrep(inner_gradient_prep, outer_pushforward_prep)
end

function DI.hvp(
    f::F,
    prep::ForwardDiffOverSomethingHVPPrep,
    backend::SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        Constant(f),
        PrepContext(inner_gradient_prep),
        Constant(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    return DI.pushforward(
        shuffled_gradient, outer_pushforward_prep, outer(backend), x, tx, new_contexts...
    )
end

function DI.hvp!(
    f::F,
    tg::NTuple,
    prep::ForwardDiffOverSomethingHVPPrep,
    backend::SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        Constant(f),
        PrepContext(inner_gradient_prep),
        Constant(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    return DI.pushforward!(
        shuffled_gradient,
        tg,
        outer_pushforward_prep,
        outer(backend),
        x,
        tx,
        new_contexts...,
    )
    return tg
end

function DI.gradient_and_hvp(
    f::F,
    prep::ForwardDiffOverSomethingHVPPrep,
    backend::SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        Constant(f),
        PrepContext(inner_gradient_prep),
        Constant(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    return DI.value_and_pushforward(
        shuffled_gradient, outer_pushforward_prep, outer(backend), x, tx, new_contexts...
    )
end

function DI.gradient_and_hvp!(
    f::F,
    grad,
    tg::NTuple,
    prep::ForwardDiffOverSomethingHVPPrep,
    backend::SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; inner_gradient_prep, outer_pushforward_prep) = prep
    rewrap = Rewrap(contexts...)
    new_contexts = (
        Constant(f),
        PrepContext(inner_gradient_prep),
        Constant(inner(backend)),
        Constant(rewrap),
        contexts...,
    )
    new_grad, _ = DI.value_and_pushforward!(
        shuffled_gradient,
        tg,
        outer_pushforward_prep,
        outer(backend),
        x,
        tx,
        new_contexts...,
    )
    return copyto!(grad, new_grad), tg
end
