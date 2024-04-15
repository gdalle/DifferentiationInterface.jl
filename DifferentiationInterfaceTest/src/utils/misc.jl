# mysimilar(x::Number) = zero(x)
mysimilar(x::AbstractArray) = similar(x)

mysimilar_random(x::Number) = randn(typeof(x))
mysimilar_random(x::AbstractArray) = map(mysimilar_random, similar(x))
