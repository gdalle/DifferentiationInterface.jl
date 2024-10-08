choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{chunksize}, x) where {chunksize} = Chunk{chunksize}()

tag_type(f, ::AutoForwardDiff{chunksize,T}, x) where {chunksize,T} = T

function tag_type(f, ::AutoForwardDiff{chunksize,Nothing}, x) where {chunksize}
    return typeof(Tag(f, eltype(x)))
end

function make_dual_similar(::Type{T}, x::Number, tx::NTuple{B}) where {T,B}
    return Dual{T}(x, tx...)
end

function make_dual_similar(::Type{T}, x, tx::NTuple{B}) where {T,B}
    return similar(x, Dual{T,eltype(x),B})
end

function make_dual(::Type{T}, x::Number, dx::Number) where {T}
    return Dual{T}(x, dx)
end

function make_dual(::Type{T}, x::Number, tx::NTuple{B}) where {T,B}
    return Dual{T}(x, tx...)
end

function make_dual(::Type{T}, x, tx::NTuple{B}) where {T,B}
    return Dual{T}.(x, tx...)
end

function make_dual!(::Type{T}, xdual, x, tx::NTuple{B}) where {T,B}
    return xdual .= Dual{T}.(x, tx...)
end

myvalue(::Type{T}, ydual::Number) where {T} = value(T, ydual)
myvalue(::Type{T}, ydual) where {T} = myvalue.(T, ydual)
myvalue!(::Type{T}, y, ydual) where {T} = y .= myvalue.(T, ydual)

myderivative(::Type{T}, ydual::Number) where {T} = extract_derivative(T, ydual)
myderivative(::Type{T}, ydual) where {T} = myderivative.(T, ydual)
myderivative!(::Type{T}, dy, ydual) where {T} = dy .= myderivative.(T, ydual)

function mypartials(::Type{T}, ::Val{B}, ydual::Number) where {T,B}
    return ntuple(Val(B)) do b
        partials(T, ydual, b)
    end
end

function mypartials(::Type{T}, ::Val{B}, ydual) where {T,B}
    return ntuple(Val(B)) do b
        partials.(T, ydual, b)
    end
end

function mypartials!(::Type{T}, ty::NTuple{B}, ydual) where {T,B}
    for b in eachindex(ty)
        ty[b] .= partials.(T, ydual, b)
    end
    return ty
end
