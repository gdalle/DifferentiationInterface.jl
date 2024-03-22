
choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{C}, x) where {C} = Chunk{C}()

tag_type(::F, x) where {F} = Tag{F,myeltype(x)}

make_dual(::Type{T}, x::Number, dx::Number) where {T} = Dual{T}(x, dx)
make_dual(::Type{T}, x, dx) where {T} = Dual{T}.(x, dx)

my_value(::Type{T}, ydual::Number) where {T} = value(T, ydual)
my_value(::Type{T}, ydual) where {T} = value.(T, ydual)

my_value!(::Type{T}, y::Number, ydual::Number) where {T} = value(T, ydual)
my_value!(::Type{T}, y, ydual) where {T} = y .= value.(T, ydual)

my_derivative(::Type{T}, ydual) where {T} = extract_derivative(T, ydual)

function my_derivative!(::Type{T}, dy::Number, ydual::Number) where {T}
    return extract_derivative(T, ydual)
end

function my_derivative!(::Type{T}, dy, ydual) where {T}
    return extract_derivative!(T, dy, ydual)
end
