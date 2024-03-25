
choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{C}, x) where {C} = Chunk{C}()

tag_type(::F, x::Number) where {F} = Tag{F,typeof(x)}
tag_type(::F, x::AbstractArray) where {F} = Tag{F,eltype(x)}
tag_type(::F, x) where {F} = Tag{F,typeof(x)}

make_dual(::Type{T}, x::Number, dx::Number) where {T} = Dual{T}(x, dx)
make_dual(::Type{T}, x::AbstractArray, dx) where {T} = Dual{T}.(x, dx)
make_dual(::Type{T}, x, dx) where {T} = fmap(Dual{T}, x, dx)

myvalue(::Type{T}, ydual::Number) where {T} = value(T, ydual)
myvalue(::Type{T}, ydual::AbstractArray) where {T} = value.(T, ydual)
myvalue(::Type{T}, ydual) where {T} = fmap(Base.Fix1(myvalue, T), ydual)

myvalue!!(::Type{T}, y::Number, ydual::Number) where {T} = value(T, ydual)
myvalue!!(::Type{T}, y::AbstractArray, ydual) where {T} = y .= value.(T, ydual)
myvalue!!(::Type{T}, y, ydual) where {T} = fmap(Base.Fix1(myvalue, T), y, ydual)

myderivative(::Type{T}, ydual::Number) where {T} = extract_derivative(T, ydual)
myderivative(::Type{T}, ydual::AbstractArray) where {T} = extract_derivative(T, ydual)
myderivative(::Type{T}, ydual) where {T} = fmap(Base.Fix1(myvalue, T), ydual)

function myderivative!!(::Type{T}, dy::Number, ydual::Number) where {T}
    return extract_derivative(T, ydual)
end

function myderivative!!(::Type{T}, dy::AbstractArray, ydual::AbstractArray) where {T}
    return extract_derivative!(T, dy, ydual)
end

function myderivative!!(::Type{T}, dy, ydual) where {T}
    return fmap(Base.Fix1(myderivative!!, T), dy, ydual)
end
