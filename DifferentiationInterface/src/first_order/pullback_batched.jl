## Docstrings

function prepare_pullback_batched end
function prepare_pullback_batched_same_point end

function value_and_pullback_batched end
function value_and_pullback_batched! end
function pullback_batched end
function pullback_batched! end

## Preparation

### Different point

function prepare_pullback_batched(f::F, backend::AbstractADType, x, dy::Batch) where {F}
    return prepare_pullback(f, backend, x, first(dy.elements))
end

function prepare_pullback_batched(f!::F, y, backend::AbstractADType, x, dy::Batch) where {F}
    return prepare_pullback(f!, y, backend, x, first(dy.elements))
end

### Same point

function prepare_pullback_batched_same_point(
    f::F, backend::AbstractADType, x, dy::Batch, extras::PullbackExtras
) where {F}
    return prepare_pullback_same_point(f, backend, x, first(dy.elements), extras)
end

function prepare_pullback_batched_same_point(
    f!::F, y, backend::AbstractADType, x, dy::Batch, extras::PullbackExtras
) where {F}
    return prepare_pullback_same_point(f!, y, backend, x, first(dy.elements), extras)
end

function prepare_pullback_batched_same_point(
    f::F, backend::AbstractADType, x, dy::Batch
) where {F}
    extras = prepare_pullback_batched(f, backend, x, dy)
    return prepare_pullback_batched_same_point(f, backend, x, dy, extras)
end

function prepare_pullback_batched_same_point(
    f!::F, y, backend::AbstractADType, x, dy::Batch
) where {F}
    extras = prepare_pullback_batched(f!, y, backend, x, dy)
    return prepare_pullback_batched_same_point(f!, y, backend, x, dy, extras)
end

## One argument

### Without extras

function value_and_pullback_batched(f::F, backend::AbstractADType, x, dy::Batch) where {F}
    return value_and_pullback_batched(
        f, backend, x, dy, prepare_pullback_batched(f, backend, x, dy)
    )
end

function value_and_pullback_batched!(
    f::F, dx, backend::AbstractADType, x, dy::Batch
) where {F}
    return value_and_pullback_batched!(
        f, dx, backend, x, dy, prepare_pullback_batched(f, backend, x, dy)
    )
end

function pullback_batched(f::F, backend::AbstractADType, x, dy::Batch) where {F}
    return pullback_batched(f, backend, x, dy, prepare_pullback_batched(f, backend, x, dy))
end

function pullback_batched!(f::F, dx, backend::AbstractADType, x, dy::Batch) where {F}
    return pullback_batched!(
        f, dx, backend, x, dy, prepare_pullback_batched(f, backend, x, dy)
    )
end

### With extras

function pullback_batched(
    f::F, backend::AbstractADType, x, dy::Batch{B}, extras::PullbackExtras
) where {F,B}
    dx_elements = ntuple(Val(B)) do b
        pullback(f, backend, x, dy.elements[b], extras)
    end
    return Batch(dx_elements)
end

function pullback_batched!(
    f::F, dx::Batch, backend::AbstractADType, x, dy::Batch, extras::PullbackExtras
) where {F}
    for b in eachindex(dx.elements, dy.elements)
        pullback!(f, dx.elements[b], backend, x, dy.elements[b], extras)
    end
    return dx
end

function value_and_pullback_batched(
    f::F, backend::AbstractADType, x, dy::Batch{B}, extras::PullbackExtras
) where {F,B}
    return f(x), pullback_batched(f, backend, x, dy, extras)
end

function value_and_pullback_batched!(
    f::F, dx::Batch, backend::AbstractADType, x, dy::Batch, extras::PullbackExtras
) where {F}
    return f(x), pullback_batched!(f, dx, backend, x, dy, extras)
end

## Two arguments

### Without extras

function value_and_pullback_batched(
    f!::F, y, backend::AbstractADType, x, dy::Batch
) where {F}
    return value_and_pullback_batched(
        f!, y, backend, x, dy, prepare_pullback_batched(f!, y, backend, x, dy)
    )
end

function value_and_pullback_batched!(
    f!::F, y, dx, backend::AbstractADType, x, dy::Batch
) where {F}
    return value_and_pullback_batched!(
        f!, y, dx, backend, x, dy, prepare_pullback_batched(f!, y, backend, x, dy)
    )
end

function pullback_batched(f!::F, y, backend::AbstractADType, x, dy::Batch) where {F}
    return pullback_batched(
        f!, y, backend, x, dy, prepare_pullback_batched(f!, y, backend, x, dy)
    )
end

function pullback_batched!(f!::F, y, dx, backend::AbstractADType, x, dy::Batch) where {F}
    return pullback_batched!(
        f!, y, dx, backend, x, dy, prepare_pullback_batched(f!, y, backend, x, dy)
    )
end

### With extras

function pullback_batched(
    f!::F, y, backend::AbstractADType, x, dy::Batch{B}, extras::PullbackExtras
) where {F,B}
    dx_elements = pullback.(Ref(f!), Ref(y), Ref(backend), Ref(x), dy.elements, Ref(extras))
    return Batch(dx_elements)
end

function pullback_batched!(
    f!::F, y, dx::Batch, backend::AbstractADType, x, dy::Batch, extras::PullbackExtras
) where {F}
    for b in eachindex(dx.elements, dy.elements)
        pullback!(f!, y, dx.elements[b], backend, x, dy.elements[b], extras)
    end
    return dx
end

function value_and_pullback_batched(
    f!::F, y, backend::AbstractADType, x, dy::Batch{B}, extras::PullbackExtras
) where {F,B}
    dx = pullback_batched(f!, y, backend, x, dy, extras)
    f!(y, x)
    return y, dx
end

function value_and_pullback_batched!(
    f!::F, y, dx::Batch, backend::AbstractADType, x, dy::Batch, extras::PullbackExtras
) where {F}
    pullback_batched!(f!, y, dx, backend, x, dy, extras)
    f!(y, x)
    return y, dx
end
