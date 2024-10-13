function DI.has_fixed_batchsize(::AutoForwardDiff{chunksize}) where {chunksize}
    return !isnothing(chunksize)
end

function DI.fixed_batchsize(::AutoForwardDiff{chunksize}) where {chunksize}
    @assert !isnothing(chunksize)
    return Val(chunksize)
end

chunk_to_val(::Chunk{C}) where {C} = Val(C)

function DI.adaptive_batchsize(::AutoForwardDiff{nothing}, a)
    chunk = Chunk(a)  # type-unstable
    return chunk_to_val(chunk)
end

function DI.threshold_batchsize(
    backend::AutoForwardDiff{chunksize1}, chunksize2::Integer
) where {chunksize1}
    chunksize = isnothing(chunksize1) ? nothing : min(chunksize1, chunksize2)
    return AutoForwardDiff(; chunksize, tag=backend.tag)
end

choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{chunksize}, x) where {chunksize} = Chunk{chunksize}()

get_tag(f, backend::AutoForwardDiff, x) = backend.tag

function get_tag(f::F, ::AutoForwardDiff{chunksize,Nothing}, x) where {F,chunksize}
    return Tag(f, eltype(x))
end

tag_type(f::F, backend::AutoForwardDiff, x) where {F} = typeof(get_tag(f, backend, x))

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
