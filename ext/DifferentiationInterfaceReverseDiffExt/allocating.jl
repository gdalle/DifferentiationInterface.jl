## Pullback

function DI.value_and_pullback!!(
    f::F,
    dx::AbstractArray,
    ::AutoReverseDiff,
    x::AbstractArray,
    dy::Number,
    extras::Nothing,
) where {F}
    y = f(x)
    gradient!(dx, f, x)
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback(
    f::F, ::AutoReverseDiff, x::AbstractArray, dy::Number, extras::Nothing
) where {F}
    y = f(x)
    dx = gradient(f, x)
    dx .*= dy
    return y, dx
end

function DI.value_and_pullback!!(
    f::F,
    dx::AbstractArray,
    ::AutoReverseDiff,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing,
) where {F}
    y = f(x)
    jac = jacobian(f, x)  # allocates
    mul!(vec(dx), transpose(jac), vec(dy))
    return y, dx
end

function DI.value_and_pullback(
    f::F, ::AutoReverseDiff, x::AbstractArray, dy::AbstractArray, extras::Nothing
) where {F}
    y = f(x)
    jac = jacobian(f, x)  # allocates
    dx = reshape(transpose(jac) * vec(dy), size(x))
    return y, dx
end

### Trick for unsupported scalar input

function DI.value_and_pullback(
    f::F, backend::AutoReverseDiff, x::Number, dy, extras::Nothing
) where {F}
    x_array = [x]
    y, dx_array = DI.value_and_pullback(f ∘ only, backend, x_array, dy)
    return y, only(dx_array)
end
