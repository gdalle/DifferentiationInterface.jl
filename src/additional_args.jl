for operator in [:value_and_pushforward, :value_and_pullback, :pushforward, :pullback]
    @eval function $operator(backend::AbstractADType, f, x, seed)
        return $operator(backend, f, x, seed, nothing)
    end
end

for operator! in [:value_and_pushforward!, :value_and_pullback!, :pushforward!, :pullback!]
    @eval function $operator!(
        storage::Union{Number,AbstractArray}, backend::AbstractADType, f, x, seed
    )
        return $operator!(storage, backend, f, x, seed, nothing)
    end
end

for operator! in [:value_and_pushforward!, :value_and_pullback!]
    @eval function $operator!(
        y::AbstractArray,
        storage::Union{Number,AbstractArray},
        backend::AbstractADType,
        f!,
        x,
        seed,
    )
        return $operator!(y, storage, backend, f!, x, seed, nothing)
    end
end

for operator in [
    :value_and_derivative,
    :value_and_multiderivative,
    :value_and_gradient,
    :value_and_jacobian,
    :derivative,
    :multiderivative,
    :gradient,
    :jacobian,
]
    @eval function $operator(backend::AbstractADType, f, x)
        return $operator(backend, f, x, nothing)
    end
    @eval function $operator(backend::AbstractADType, f, x, extras)
        return $operator(backend, f, x, extras, autodiff_mode(backend))
    end
end

for operator! in [
    :value_and_multiderivative!,
    :value_and_gradient!,
    :value_and_jacobian!,
    :multiderivative!,
    :gradient!,
    :jacobian!,
]
    @eval function $operator!(storage::AbstractArray, backend::AbstractADType, f, x)
        return $operator!(storage, backend, f, x, nothing)
    end
    @eval function $operator!(storage::AbstractArray, backend::AbstractADType, f, x, extras)
        return $operator!(storage, backend, f, x, extras, autodiff_mode(backend))
    end
end

for operator! in [:value_and_multiderivative!, :value_and_jacobian!]
    @eval function $operator!(
        y::AbstractArray, storage::AbstractArray, backend::AbstractADType, f!, x
    )
        return $operator!(y, storage, backend, f!, x, nothing)
    end
    @eval function $operator!(
        y::AbstractArray, storage::AbstractArray, backend::AbstractADType, f!, x, extras
    )
        return $operator!(y, storage, backend, f!, x, extras, autodiff_mode(backend))
    end
end
