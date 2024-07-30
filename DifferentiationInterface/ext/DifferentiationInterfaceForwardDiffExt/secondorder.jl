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

## Standard

function DI.prepare_hvp(f::F, backend::SecondOrder{<:AutoForwardDiff}, x, dx) where {F}
    tagged_outer_backend = tag_backend_hvp(f, outer(backend), x)
    T = tag_type(f, tagged_outer_backend, x)
    xdual = make_dual(T, x, dx)
    gradient_extras = DI.prepare_gradient(f, inner(backend), xdual)
    inner_gradient = DI.Gradient(f, inner(backend), gradient_extras)
    outer_pushforward_extras = DI.prepare_pushforward(
        inner_gradient, tagged_outer_backend, x, dx
    )
    return ForwardDiffOverSomethingHVPExtras(
        tagged_outer_backend, inner_gradient, outer_pushforward_extras
    )
end

function DI.hvp(
    f, ::SecondOrder{<:AutoForwardDiff}, x, dx, extras::ForwardDiffOverSomethingHVPExtras
)
    @compat (; tagged_outer_backend, inner_gradient, outer_pushforward_extras) = extras
    return DI.pushforward(
        inner_gradient, tagged_outer_backend, x, dx, outer_pushforward_extras
    )
end

function DI.hvp!(
    f,
    dg,
    ::SecondOrder{<:AutoForwardDiff},
    x,
    dx,
    extras::ForwardDiffOverSomethingHVPExtras,
)
    @compat (; tagged_outer_backend, inner_gradient, outer_pushforward_extras) = extras
    return DI.pushforward!(
        inner_gradient, dg, tagged_outer_backend, x, dx, outer_pushforward_extras
    )
end

## Batched

function DI.prepare_hvp_batched(
    f::F, backend::SecondOrder{<:AutoForwardDiff}, x, dx::Batch
) where {F}
    tagged_outer_backend = tag_backend_hvp(f, outer(backend), x)
    T = tag_type(f, tagged_outer_backend, x)
    xdual = make_dual(T, x, dx)
    gradient_extras = DI.prepare_gradient(f, inner(backend), xdual)
    inner_gradient = DI.Gradient(f, inner(backend), gradient_extras)
    outer_pushforward_extras = DI.prepare_pushforward_batched(
        inner_gradient, tagged_outer_backend, x, dx
    )
    return ForwardDiffOverSomethingHVPExtras(
        tagged_outer_backend, inner_gradient, outer_pushforward_extras
    )
end

function DI.hvp_batched(
    f,
    ::SecondOrder{<:AutoForwardDiff},
    x,
    dx::Batch,
    extras::ForwardDiffOverSomethingHVPExtras,
)
    @compat (; tagged_outer_backend, inner_gradient, outer_pushforward_extras) = extras
    return DI.pushforward_batched(
        inner_gradient, tagged_outer_backend, x, dx, outer_pushforward_extras
    )
end

function DI.hvp_batched!(
    f,
    dg::Batch,
    ::SecondOrder{<:AutoForwardDiff},
    x,
    dx::Batch,
    extras::ForwardDiffOverSomethingHVPExtras,
)
    @compat (; tagged_outer_backend, inner_gradient, outer_pushforward_extras) = extras
    DI.pushforward_batched!(
        inner_gradient, dg, tagged_outer_backend, x, dx, outer_pushforward_extras
    )
    return dg
end
