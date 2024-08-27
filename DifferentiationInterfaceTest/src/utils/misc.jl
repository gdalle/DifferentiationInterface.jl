mysimilar(x::Number) = one(x)
mysimilar(x::AbstractArray) = similar(x)
mysimilar(x) = deepcopy(x)
mysimilar(x::Tangents) = Tangents(map(mysimilar, x.d))

mycopy_random(rng::AbstractRNG, x::Number) = randn(rng, typeof(x))
mycopy_random(rng::AbstractRNG, x::AbstractArray) = map(Base.Fix1(mycopy_random, rng), x)
mycopy_random(rng::AbstractRNG, x) = deepcopy(x)

function mycopy_random(rng::AbstractRNG, x::Tangents)
    return Tangents(map(Base.Fix1(mycopy_random, rng), x.d))
end

mycopy_random(x) = mycopy_random(default_rng(), x)

mysize(x::Number) = size(x)
mysize(x::AbstractArray) = size(x)
mysize(x) = missing
