function initialize!(xt::TPS, x::Number, dx::Number)
    xt[0] = x
    xt[1] = dx
    return xt
end

function initialize!(xt::AbstractArray{<:TPS}, x::AbstractArray, dx::AbstractArray)
    for i in eachindex(xt, x, dx)
        initialize!(xt[i], x[i], dx[i])
    end
    return xt
end
