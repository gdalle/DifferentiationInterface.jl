choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{C}, x) where {C} = Chunk{C}()

tag_type(::F, x::Number) where {F} = Tag{F,typeof(x)}
tag_type(::F, x::AbstractArray) where {F} = Tag{F,eltype(x)}

make_dual(::Type{T}, x::Number, dx) where {T} = Dual{T}(x, dx)
make_dual(::Type{T}, x::AbstractArray, dx) where {T} = Dual{T}.(x, dx)

function make_dual!(::Type{T}, xdual, x::AbstractArray, dx) where {T}
    for i in eachindex(xdual, x, dx)
        xdual[i] = Dual{T}(x[i], dx[i])
    end
    return nothing
end

myvalue(::Type{T}, ydual::Number) where {T} = value(T, ydual)
myvalue(::Type{T}, ydual::AbstractArray) where {T} = value.(T, ydual)

function myvalue!(::Type{T}, y::AbstractArray, ydual) where {T}
    for i in eachindex(y, ydual)
        y[i] = value(T, ydual[i])
    end
    return nothing
end

myderivative(::Type{T}, ydual::Number) where {T} = extract_derivative(T, ydual)
myderivative(::Type{T}, ydual::AbstractArray) where {T} = extract_derivative(T, ydual)

function myderivative!(::Type{T}, dy, ydual::AbstractArray) where {T}
    extract_derivative!(T, dy, ydual)
    return nothing
end

function myvalueandderivative!(::Type{T}, y, dy, ydual::AbstractArray) where {T}
    for i in eachindex(y, dy, ydual)
        y[i] = value(T, ydual[i])
        dy[i] = extract_derivative(T, ydual[i])
    end
    return nothing
end
