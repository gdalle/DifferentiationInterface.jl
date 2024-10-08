struct ForwardDiffOverSomethingHVPWrapper{F}
    f::F
end

"""
    tag_backend_hvp(f, ::AutoForwardDiff, x)

Return a new `AutoForwardDiff` backend with a fixed tag linked to `f`, so that we know how to prepare the inner gradient of the HVP without depending on what that gradient closure looks like.
"""
function tag_backend_hvp(f::F, ::AutoForwardDiff{chunksize,Nothing}, x) where {F,chunksize}
    return AutoForwardDiff(;
        chunksize=chunksize,
        tag=ForwardDiff.Tag(ForwardDiffOverSomethingHVPWrapper(f), eltype(x)),
    )
end

function tag_backend_hvp(f, backend::AutoForwardDiff, x)
    return backend
end

struct ForwardDiffOverSomethingHVPPrep{B<:AutoForwardDiff,G,E<:PushforwardPrep} <: HVPPrep
    tagged_outer_backend::B
    inner_gradient::G
    outer_pushforward_prep::E
end

function DI.prepare_hvp(
    f::F,
    backend::SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    rewrap = Rewrap(contexts...)
    tagged_outer_backend = tag_backend_hvp(f, outer(backend), x)
    T = tag_type(f, tagged_outer_backend, x)
    xdual = make_dual(T, x, tx)
    gradient_prep = DI.prepare_gradient(f, inner(backend), xdual, contexts...)
    function inner_gradient(x, unannotated_contexts...)
        annotated_contexts = rewrap(unannotated_contexts...)
        return DI.gradient(f, gradient_prep, inner(backend), x, annotated_contexts...)
    end
    outer_pushforward_prep = DI.prepare_pushforward(
        inner_gradient, tagged_outer_backend, x, tx, contexts...
    )
    return ForwardDiffOverSomethingHVPPrep(
        tagged_outer_backend, inner_gradient, outer_pushforward_prep
    )
end

function DI.hvp(
    f::F,
    prep::ForwardDiffOverSomethingHVPPrep,
    ::SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; tagged_outer_backend, inner_gradient, outer_pushforward_prep) = prep
    return DI.pushforward(
        inner_gradient, outer_pushforward_prep, tagged_outer_backend, x, tx, contexts...
    )
end

function DI.hvp!(
    f::F,
    tg::NTuple,
    prep::ForwardDiffOverSomethingHVPPrep,
    ::SecondOrder{<:AutoForwardDiff},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    (; tagged_outer_backend, inner_gradient, outer_pushforward_prep) = prep
    DI.pushforward!(
        inner_gradient, tg, outer_pushforward_prep, tagged_outer_backend, x, tx, contexts...
    )
    return tg
end
