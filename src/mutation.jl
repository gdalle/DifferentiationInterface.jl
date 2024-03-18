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

Return the mutation behavior of a backend.
"""
mutation_behavior(::AbstractADType) = MutationSupported()
