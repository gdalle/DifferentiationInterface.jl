function DI.value_and_pullback!(
    y::AbstractArray,
    dx::AbstractArray,
    ::AutoReverseDiff,
    f!,
    x::AbstractArray,
    dy::AbstractArray,
    extras::Nothing=nothing,
)
    res = DiffResults.DiffResult(y, similar(dy, length(y), length(x)))
    res = jacobian!(res, f!, y, x)
    jac = DiffResults.jacobian(res)
    mul!(vec(dx), transpose(jac), vec(dy))
    return DiffResults.value(res), dx
end

function DI.value_and_pullback!(
    y::AbstractArray,
    _dx::Number,
    backend::AutoReverseDiff,
    f!,
    x::Number,
    dy,
    extras::Nothing=nothing,
)
    x_array = [x]
    dx_array = similar(x_array)
    f!_only(_y::AbstractArray, _x_array) = f!(_y, only(_x_array))
    y, dx_array = DI.value_and_pullback!(y, dx_array, backend, f!_only, x_array, dy, extras)
    return y, only(dx_array)
end
