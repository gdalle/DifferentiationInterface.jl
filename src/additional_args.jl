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

for operator in [:gradient_and_hessian_vector_product]
    @eval function $operator(backend::AbstractADType, f, x, v)
        return $operator(backend::AbstractADType, f, x, v, nothing)
    end
end

for operator! in [:gradient_and_hessian_vector_product!]
    @eval function $operator!(
        grad::AbstractArray, hvp::AbstractArray, backend::AbstractADType, f, x, v
    )
        return $operator!(grad, hvp, backend, f, x, v, nothing)
    end
end

for operator in [:value_and_gradient_and_hessian, :hessian]
    @eval function $operator(backend::AbstractADType, f, x)
        return $operator(backend, f, x, nothing)
    end
end

for operator! in [:value_and_gradient_and_hessian!]
    @eval function $operator!(
        grad::AbstractArray, hess::AbstractMatrix, backend::AbstractADType, f, x
    )
        return $operator!(grad, hess, backend, f, x, nothing)
    end
end

for operator! in [:hessian!]
    @eval function $operator!(hess::AbstractMatrix, backend::AbstractADType, f, x)
        return $operator!(hess, backend, f, x, nothing)
    end
end
