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

struct ForwardDiffOverSomethingHVPExtras{
    B<:AutoForwardDiff,G<:DI.Gradient,E<:PushforwardExtras
} <: HVPExtras
    tagged_outer_backend::B
    inner_gradient::G
    outer_pushforward_extras::E
end

function DI.prepare_hvp(
    f::F, backend::SecondOrder{<:AutoForwardDiff}, x, tx::Tangents
) where {F}
    tagged_outer_backend = tag_backend_hvp(f, outer(backend), x)
    T = tag_type(f, tagged_outer_backend, x)
    xdual = make_dual(T, x, tx)
    gradient_extras = DI.prepare_gradient(f, inner(backend), xdual)
    inner_gradient = DI.Gradient(f, inner(backend), gradient_extras)
    outer_pushforward_extras = DI.prepare_pushforward(
        inner_gradient, tagged_outer_backend, x, tx
    )
    return ForwardDiffOverSomethingHVPExtras(
        tagged_outer_backend, inner_gradient, outer_pushforward_extras
    )
end

function DI.hvp(
    f,
    ::SecondOrder{<:AutoForwardDiff},
    x,
    tx::Tangents,
    extras::ForwardDiffOverSomethingHVPExtras,
)
    @compat (; tagged_outer_backend, inner_gradient, outer_pushforward_extras) = extras
    return DI.pushforward(
        inner_gradient, tagged_outer_backend, x, tx, outer_pushforward_extras
    )
end

function DI.hvp!(
    f,
    tg::Tangents,
    ::SecondOrder{<:AutoForwardDiff},
    x,
    tx::Tangents,
    extras::ForwardDiffOverSomethingHVPExtras,
)
    @compat (; tagged_outer_backend, inner_gradient, outer_pushforward_extras) = extras
    DI.pushforward!(
        inner_gradient, tg, tagged_outer_backend, x, tx, outer_pushforward_extras
    )
    return tg
end
