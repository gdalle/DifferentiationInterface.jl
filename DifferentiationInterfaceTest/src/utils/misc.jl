mysimilar(x::Number) = one(x)
mysimilar(x::AbstractArray) = similar(x)
mysimilar(x) = deepcopy(x)
mysimilar(x::Batch) = Batch(map(mysimilar, x.elements))

mycopy_random(rng::AbstractRNG, x::Number) = randn(rng, typeof(x))
mycopy_random(rng::AbstractRNG, x::AbstractArray) = map(Base.Fix1(mycopy_random_aux, rng), x)
mycopy_random(rng::AbstractRNG, x) = deepcopy(x)

function mycopy_random(rng::AbstractRNG, x::Batch)
    return Batch(map(Base.Fix1(mycopy_random, rng), x.elements))
end

mycopy_random(x) = mycopy_random(default_rng(), x)

mysize(x::Number) = size(x)
mysize(x::AbstractArray) = size(x)
mysize(x) = missing
