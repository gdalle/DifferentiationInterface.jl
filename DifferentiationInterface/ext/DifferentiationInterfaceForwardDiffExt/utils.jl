choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{C}, x) where {C} = Chunk{min(length(x), C)}()

tag_type(f, ::AutoForwardDiff{C,T}, x) where {C,T} = T
tag_type(f, ::AutoForwardDiff{C,Nothing}, x) where {C} = typeof(Tag(f, eltype(x)))

function make_dual_similar(::Type{T}, x::Number, tx::Tangents{B,<:Number}) where {T,B}
    return Dual{T}(x, dx.d)
end

function make_dual_similar(::Type{T}, x, tx::Tangents{B}) where {T,B}
    return similar(x, Dual{T,eltype(x),B})
end

function make_dual(::Type{T}, x::Number, dx::Tangents{B,<:Number}) where {T,B}
    return Dual{T}(x, dx.d...)
end

function make_dual(::Type{T}, x, dx::Tangents{B}) where {T,B}
    return Dual{T}.(x, dx.d...)
end

function make_dual!(::Type{T}, xdual, x, dx::Tangents{B}) where {T,B}
    return xdual .= Dual{T}.(x, dx.d...)
end

myvalue(::Type{T}, ydual::Dual{T}) where {T} = value(T, ydual)
myvalue(::Type{T}, ydual) where {T} = myvalue.(T, ydual)
myvalue!(::Type{T}, y, ydual) where {T} = y .= myvalue.(T, ydual)

myderivative(::Type{T}, ydual::Dual{T}) where {T} = extract_derivative(T, ydual)
myderivative(::Type{T}, ydual) where {T} = myderivative.(T, ydual)
myderivative!(::Type{T}, dy, ydual) where {T} = dy .= myderivative.(T, ydual)

function mypartials(::Type{T}, ::Val{B}, ydual::Dual) where {T,B}
    elements = partials(T, ydual).values
    return Tangents(elements...)
end

function mypartials(::Type{T}, ::Val{B}, ydual) where {T,B}
    elements = ntuple(Val(B)) do b
        partials.(T, ydual, b)
    end
    return Tangents(elements...)
end

function mypartials!(::Type{T}, dy::Tangents{B}, ydual) where {T,B}
    for b in eachindex(dy.d)
        dy.d[b] .= partials.(T, ydual, b)
    end
    return dy
end
