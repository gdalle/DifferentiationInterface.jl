choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{C}, x) where {C} = Chunk{min(length(x), C)}()

tag_type(f, ::AutoForwardDiff{C,T}, x) where {C,T} = T
tag_type(f, ::AutoForwardDiff{C,Nothing}, x) where {C} = Tag{typeof(f),eltype(x)}

make_dual_similar(::Type{T}, x::Number, dx::Number) where {T} = Dual{T}(x, dx)
make_dual_similar(::Type{T}, x, dx) where {T} = similar(x, Dual{T,eltype(x),1})

function make_dual_similar(::Type{T}, x::Number, dx::Batch{B,<:Number}) where {T,B}
    return Dual{T}(x, dx.elements...)
end

function make_dual_similar(::Type{T}, x, dx::Batch{B}) where {T,B}
    return similar(x, Dual{T,eltype(x),B})
end

make_dual(::Type{T}, x::Number, dx::Number) where {T} = Dual{T}(x, dx)
make_dual!(::Type{T}, xdual, x, dx) where {T} = map!(Dual{T}, xdual, x, dx)

function make_dual(::Type{T}, x::Number, dx::Batch{B,<:Number}) where {T,B}
    return Dual{T}(x, dx.elements...)
end

function make_dual!(::Type{T}, xdual, x, dx::Batch{B}) where {T,B}
    return map!(Dual{T}, xdual, x, dx.elements...)
end

myvalue(::Type{T}, ydual::Dual{T}) where {T} = value(T, ydual)
myvalue(::Type{T}, ydual) where {T} = map(Fix1(myvalue, T), ydual)

function myvalue!(::Type{T}, y, ydual) where {T}
    return map!(Fix1(myvalue, T), y, ydual)
end

myderivative(::Type{T}, ydual::Dual{T}) where {T} = extract_derivative(T, ydual)
myderivative(::Type{T}, ydual) where {T} = map(Fix1(myderivative, T), ydual)

function myderivative!(::Type{T}, dy, ydual) where {T}
    return map!(Fix1(myderivative, T), dy, ydual)
end

mypartials(::Type{T}, ydual::Dual{T}) where {T} = Batch(partials(ydual).values)

function mypartials(::Type{T}, ydual) where {T}
    B = npartials(eltype(ydual))
    elements = ntuple(Val(B)) do b
        map(Fix2(partials, b), ydual)
    end
    return Batch(elements)
end

function mypartials!(::Type{T}, dy::Batch{B}, ydual) where {T,B}
    for b in eachindex(dy.elements)
        for i in eachindex(dy.elements[b], ydual)
            dy.elements[b][i] = partials(ydual[i], b)
        end
    end
    return dy
end
