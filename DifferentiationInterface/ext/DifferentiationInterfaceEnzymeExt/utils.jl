struct AutoDeferredEnzyme{M} <: ADTypes.AbstractADType
    mode::M
end

ADTypes.mode(backend::AutoDeferredEnzyme) = ADTypes.mode(AutoEnzyme(backend.mode))

function DI.nested(backend::AutoEnzyme{M}) where {M}
    return AutoDeferredEnzyme{M}(backend.mode)
end

const AnyAutoEnzyme{M,constant_function} = Union{
    AutoEnzyme{M,constant_function},AutoDeferredEnzyme{M,constant_function}
}

# forward mode if possible
forward_mode(backend::AnyAutoEnzyme{<:Mode}) = backend.mode
forward_mode(::AnyAutoEnzyme{Nothing}) = Forward

# reverse mode if possible
reverse_mode(backend::AnyAutoEnzyme{<:Mode}) = backend.mode
reverse_mode(::AnyAutoEnzyme{Nothing}) = Reverse

DI.check_available(::AutoEnzyme) = true

# until https://github.com/EnzymeAD/Enzyme.jl/pull/1545 is merged
DI.pick_batchsize(::AnyAutoEnzyme, dimension::Integer) = min(dimension, 16)

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DI.basis(::AnyAutoEnzyme, a::AbstractArray{T}, i::CartesianIndex) where {T}
    b = zero(a)
    b[i] = one(T)
    return b
end

get_f_and_df(f, ::AnyAutoEnzyme) = Const(f)
