## Docstrings

function prepare_hvp_batched end
function prepare_hvp_batched_same_point end

function hvp_batched end
function hvp_batched! end

## Preparation

### Different point

function prepare_hvp_batched(f::F, backend::AbstractADType, x, dx::Batch) where {F}
    return prepare_hvp_batched(f, SecondOrder(backend, backend), x, dx)
end

function prepare_hvp_batched(f::F, backend::SecondOrder, x, dx::Batch) where {F}
    return prepare_hvp_batched_aux(f, backend, x, dx, hvp_mode(backend))
end

function prepare_hvp_batched_aux(
    f::F, backend::SecondOrder, x, dx::Batch, ::ForwardOverForward
) where {F}
    # batched pushforward of gradient
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pushforward_extras = prepare_pushforward_batched(
        inner_gradient, outer(backend), x, dx
    )
    return ForwardOverForwardHVPExtras(inner_gradient, outer_pushforward_extras)
end

function prepare_hvp_batched_aux(
    f::F, backend::SecondOrder, x, dx::Batch, ::ForwardOverReverse
) where {F}
    # batched pushforward of gradient
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pushforward_extras = prepare_pushforward_batched(
        inner_gradient, outer(backend), x, dx
    )
    return ForwardOverReverseHVPExtras(inner_gradient, outer_pushforward_extras)
end

function prepare_hvp_batched_aux(
    f::F, backend::SecondOrder, x, dx::Batch, ::ReverseOverForward
) where {F}
    # TODO: batched version replacing the outer gradient with a pullback
    return prepare_hvp_aux(f, backend, x, first(dx.d), ReverseOverForward())
end

function prepare_hvp_batched_aux(
    f::F, backend::SecondOrder, x, dx::Batch, ::ReverseOverReverse
) where {F}
    # batched pullback of gradient
    inner_gradient = Gradient(f, nested(inner(backend)))
    outer_pullback_extras = prepare_pullback_batched(inner_gradient, outer(backend), x, dx)
    return ReverseOverReverseHVPExtras(inner_gradient, outer_pullback_extras)
end

### Same point

function prepare_hvp_batched_same_point(
    f::F, backend::AbstractADType, x, dx::Batch, extras::HVPExtras
) where {F}
    return extras
end

function prepare_hvp_batched_same_point(
    f::F, backend::AbstractADType, x, dx::Batch
) where {F}
    extras = prepare_hvp_batched(f, backend, x, dx)
    return prepare_hvp_batched_same_point(f, backend, x, dx, extras)
end

## One argument

function hvp_batched(
    f::F, backend::AbstractADType, x, dx::Batch, extras::HVPExtras
) where {F}
    return hvp_batched(f, SecondOrder(backend, backend), x, dx, extras)
end

function hvp_batched(
    f::F, backend::SecondOrder, x, dx::Batch, extras::ForwardOverForwardHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward_batched(
        inner_gradient, outer(backend), x, dx, outer_pushforward_extras
    )
end

function hvp_batched(
    f::F, backend::SecondOrder, x, dx::Batch, extras::ForwardOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward_batched(
        inner_gradient, outer(backend), x, dx, outer_pushforward_extras
    )
end

function hvp_batched(
    f::F, backend::SecondOrder, x, dx::Batch, extras::ReverseOverForwardHVPExtras
) where {F}
    dg_elements = hvp.(Ref(f), Ref(backend), Ref(x), dx.d, Ref(extras))
    return Batch(dg_elements)
end

function hvp_batched(
    f::F, backend::SecondOrder, x, dx::Batch, extras::ReverseOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback_batched(inner_gradient, outer(backend), x, dx, outer_pullback_extras)
end

function hvp_batched!(
    f::F, dg::Batch, backend::AbstractADType, x, dx::Batch, extras::HVPExtras
) where {F}
    return hvp_batched!(f, dg, SecondOrder(backend, backend), x, dx, extras)
end

function hvp_batched!(
    f::F, dg::Batch, backend::SecondOrder, x, dx::Batch, extras::ForwardOverForwardHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward_batched!(
        inner_gradient, dg, outer(backend), x, dx, outer_pushforward_extras
    )
end

function hvp_batched!(
    f::F, dg::Batch, backend::SecondOrder, x, dx::Batch, extras::ForwardOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pushforward_extras) = extras
    return pushforward_batched!(
        inner_gradient, dg, outer(backend), x, dx, outer_pushforward_extras
    )
end

function hvp_batched!(
    f::F, dg::Batch, backend::SecondOrder, x, dx::Batch, extras::ReverseOverForwardHVPExtras
) where {F}
    for b in eachindex(dg.d, dx.d)
        hvp!(f, dg.d[b], backend, x, dx.d[b], extras)
    end
    return dg
end

function hvp_batched!(
    f::F, dg::Batch, backend::SecondOrder, x, dx::Batch, extras::ReverseOverReverseHVPExtras
) where {F}
    @compat (; inner_gradient, outer_pullback_extras) = extras
    return pullback_batched!(
        inner_gradient, dg, outer(backend), x, dx, outer_pullback_extras
    )
end
