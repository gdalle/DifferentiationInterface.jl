mysimilar(x::AbstractArray) = similar(x)

mycopy_random(x::Number) = randn(typeof(x))
mycopy_random(x::AbstractArray) = map(mycopy_random, x)
