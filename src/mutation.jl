abstract type MutationBehavior end

"""
    MutationSupported

Trait identifying backends that support mutating functions `f!(y, x)`.
"""
struct MutationSupported <: MutationBehavior end

"""
    MutationNotSupported

Trait identifying backends that do not support mutating functions `f!(y, x)`.
"""
struct MutationNotSupported <: MutationBehavior end

"""
    mutation_behavior(backend)

Return the mutation behavior of a backend in a statically predictable way.

# Note

This is different from [`supports_mutation`](@ref), which performs an actual call to `jacobian!`.
"""
mutation_behavior(::AbstractADType) = MutationSupported()
