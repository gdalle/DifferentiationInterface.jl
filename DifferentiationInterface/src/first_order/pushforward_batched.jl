## Docstrings

function prepare_pushforward_batched end
function prepare_pushforward_batched_same_point end

function value_and_pushforward_batched end
function value_and_pushforward_batched! end
function pushforward_batched end
function pushforward_batched! end

## Preparation

function prepare_pushforward_batched(f::F, backend::AbstractADType, x, dx::Batch) where {F}
    return prepare_pushforward(f, backend, x, first(dx.elements))
end

function prepare_pushforward_batched(
    f!::F, y, backend::AbstractADType, x, dx::Batch
) where {F}
    return prepare_pushforward(f!, y, backend, x, first(dx.elements))
end

## One argument

function pushforward_batched(
    f::F, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    dy_elements = pushforward.(Ref(f), Ref(backend), Ref(x), dx.elements, Ref(extras))
    return Batch(dy_elements)
end

function pushforward_batched!(
    f::F, dy::Batch, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    for b in eachindex(dy.elements, dx.elements)
        pushforward!(f, dy.elements[b], backend, x, dx.elements[b], extras)
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
        pushforward.(Ref(f!), Ref(y), Ref(backend), Ref(x), dx.elements, Ref(extras))
    return Batch(dy_elements)
end

function pushforward_batched!(
    f!::F, y, dy::Batch, backend::AbstractADType, x, dx::Batch, extras::PushforwardExtras
) where {F}
    for b in eachindex(dy.elements, dx.elements)
        pushforward!(f!, y, dy.elements[b], backend, x, dx.elements[b], extras)
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
