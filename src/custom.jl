for utility in [
    :value_and_derivative,
    :value_and_multiderivative,
    :value_and_gradient,
    :value_and_jacobian,
    :derivative,
    :multiderivative,
    :gradient,
    :jacobian,
]
    @eval $utility(backend::AbstractADType, f, x::Union{Number,AbstractArray}) =
        $utility(CustomImplem(), backend, f, x)
end

for utility! in [
    :value_and_multiderivative!,
    :value_and_gradient!,
    :value_and_jacobian!,
    :multiderivative!,
    :gradient!,
    :jacobian!,
]
    @eval $utility!(
        storage::Union{Number,AbstractArray},
        backend::AbstractADType,
        f,
        x::Union{Number,AbstractArray},
    ) = $utility!(CustomImplem(), storage, backend, f, x)
end
