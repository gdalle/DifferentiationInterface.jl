mysimilar(x::Number) = zero(x)
mysimilar(x::AbstractArray) = similar(x)
mysimilar(x) = fmap(mysimilar, x)
mysimilar(x::Batch) = Batch(map(mysimilar, x.elements))

mycopy_random(x) = mycopy_random(default_rng(), x)
mycopy_random(rng::AbstractRNG, x::Number) = randn(rng, typeof(x))
mycopy_random(rng::AbstractRNG, x::AbstractArray) = map(Base.Fix1(mycopy_random, rng), x)
mycopy_random(rng::AbstractRNG, x) = fmap(Base.Fix1(mycopy_random, rng), x)

function mycopy_random(rng::AbstractRNG, x::Batch)
    return Batch(map(Base.Fix1(mycopy_random, rng), x.elements))
end

mysize(x::Number) = size(x)
mysize(x::AbstractArray) = size(x)
mysize(x) = missing
