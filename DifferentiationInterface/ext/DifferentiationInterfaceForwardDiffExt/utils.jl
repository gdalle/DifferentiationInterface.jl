choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{C}, x) where {C} = Chunk{C}()

tag_type(f, ::AutoForwardDiff{C,T}, x) where {C,T} = T
tag_type(f, ::AutoForwardDiff{C,Nothing}, x) where {C} = Tag{typeof(f),eltype(x)}

make_dual(::Type{T}, x::Number, dx) where {T} = Dual{T}(x, dx)
make_dual(::Type{T}, x, dx) where {T} = Dual{T}.(x, dx)  # TODO: map causes Enzyme to fail

make_dual_similar(::Type{T}, x::Number) where {T} = Dual{T}(x, x)
make_dual_similar(::Type{T}, x) where {T} = similar(x, Dual{T})

make_dual!(::Type{T}, xdual, x, dx) where {T} = map!(Dual{T}, xdual, x, dx)

myvalue(::Type{T}, ydual::Number) where {T} = value(T, ydual)
myvalue(::Type{T}, ydual) where {T} = map(Fix1(value, T), ydual)

myvalue!(::Type{T}, y, ydual) where {T} = map!(Fix1(value, T), y, ydual)

myderivative(::Type{T}, ydual::Number) where {T} = extract_derivative(T, ydual)
myderivative(::Type{T}, ydual) where {T} = map(Fix1(extract_derivative, T), ydual)

function myderivative!(::Type{T}, dy, ydual) where {T}
    return map!(Fix1(extract_derivative, T), dy, ydual)
end
