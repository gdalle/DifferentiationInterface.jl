similar_random(z::Number) = randn(eltype(z))

function similar_random(z::AbstractArray)
    zz = similar(z)
    zz .= randn(eltype(zz), size(zz))
    return zz
end
