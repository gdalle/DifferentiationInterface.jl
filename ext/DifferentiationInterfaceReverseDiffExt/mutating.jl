## Pullback

function DI.value_and_pullback!!(
    f!::F,
    y::AbstractArray,
    dx::AbstractArray,
    ::AutoReverseDiff,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing,
) where {F}
    jac = jacobian(f!, y, x)
    mul!(vec(dx), transpose(jac), vec(dy))
    return y, dx
end

### Trick for unsupported scalar input

function DI.value_and_pullback!!(
    f!::F,
    y::AbstractArray,
    dx::Number,
    backend::AutoReverseDiff,
    x::Number,
    dy::AbstractArray,
    extras::Nothing,
) where {F}
    x_array = [x]
    dx_array = similar(x_array)
    f!_only(_y::AbstractArray, _x_array) = f!(_y, only(_x_array))
    y, dx_array = DI.value_and_pullback!!(f!_only, y, dx_array, backend, x_array, dy)
    return y, only(dx_array)
end
