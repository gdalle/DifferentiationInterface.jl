function value_and_gradient!(
    dx::AbstractArray, backend::AbstractForwardBackend, f, x::AbstractArray
)
    y = f(x)
    for i in eachindex(x)
        dx_i = basisarray(backend, x, i)
        dx[i] = pushforward!(y, backend, f, x, dx_i)
    end
    return y, dx
end

function value_and_gradient!(
    dx::AbstractArray, backend::AbstractReverseBackend, f, x::AbstractArray
)
    y = f(x)
    return y, pullback!(dx, backend, f, x, one(y))
end
