## Docstrings

function prepare_pushforward_batched end
function prepare_pushforward_batched_same_point end

function value_and_pushforward_batched end
function value_and_pushforward_batched! end
function pushforward_batched end
function pushforward_batched! end

## Preparation

### Different point

function prepare_pushforward_batched(f::F, backend::AbstractADType, x, dx::Batch) where {F}
    return prepare_pushforward(f, backend, x, first(dx.d))
end

function prepare_pushforward_batched(
    f!::F, y, backend::AbstractADType, x, dx::Batch
) where {F}
    return prepare_pushforward(f!, y, backend, x, first(dx.d))
end

### Same point

function prepare_pushforward_batched_same_point(
    f::F, backend::AbstractADType, x, dx::Batch
) where {F}
    extras = prepare_pushforward_batched(f, backend, x, dx)
    return prepare_pushforward_batched_same_point(f, backend, x, dx, extras)
end

function prepare_pushforward_batched_same_point(
    f!::F, y, backend::AbstractADType, x, dx::Batch
) where {F}
    extras = prepare_pushforward_batched(f!, y, backend, x, dx)
    return prepare_pushforward_batched_same_point(f!, y, backend, x, dx, extras)
end

function prepare_pushforward_batched_same_point(
    f::F, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    return prepare_pushforward_same_point(f, backend, x, first(dx.d), extras)
end

function prepare_pushforward_batched_same_point(
    f!::F, y, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    return prepare_pushforward_same_point(f!, y, backend, x, first(dx.d), extras)
end

## One argument

function pushforward_batched(
    f::F, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    dy_elements = pushforward.(Ref(f), Ref(backend), Ref(x), dx.d, Ref(extras))
    return Batch(dy_elements)
end

function pushforward_batched!(
    f::F, dy::Batch, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    for b in eachindex(dy.d, dx.d)
        pushforward!(f, dy.d[b], backend, x, dx.d[b], extras)
    end
    return dy
end

function value_and_pushforward_batched(
    f::F, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    return f(x), pushforward_batched(f, backend, x, dx, extras)
end

function value_and_pushforward_batched!(
    f::F, dy::Batch, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    return f(x), pushforward_batched!(f, dy, backend, x, dx, extras)
end

## Two arguments

function pushforward_batched(
    f!::F, y, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    dy_elements =
        pushforward.(Ref(f!), Ref(y), Ref(backend), Ref(x), dx.d, Ref(extras))
    return Batch(dy_elements)
end

function pushforward_batched!(
    f!::F, y, dy::Batch, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    for b in eachindex(dy.d, dx.d)
        pushforward!(f!, y, dy.d[b], backend, x, dx.d[b], extras)
    end
    return dy
end

function value_and_pushforward_batched(
    f!::F, y, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    dy = pushforward_batched(f!, y, backend, x, dx, extras)
    f!(y, x)
    return y, dy
end

function value_and_pushforward_batched!(
    f!::F, y, dy::Batch, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    pushforward_batched!(f!, y, dy, backend, x, dx, extras)
    f!(y, x)
    return y, dy
end
