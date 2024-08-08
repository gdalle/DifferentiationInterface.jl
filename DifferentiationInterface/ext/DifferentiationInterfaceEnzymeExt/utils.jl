struct AutoDeferredEnzyme{M,A} <: ADTypes.AbstractADType
    mode::M
end

function ADTypes.mode(backend::AutoDeferredEnzyme{M,A}) where {M,A}
    return ADTypes.mode(AutoEnzyme{M,A}(backend.mode))
end

function DI.nested(backend::AutoEnzyme{M,A}) where {M,A}
    return AutoDeferredEnzyme(; mode=backend.mode, function_annotation=A)
end

const AnyAutoEnzyme{M,A} = Union{AutoEnzyme{M,A},AutoDeferredEnzyme{M,A}}

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

function get_f_and_df(f, ::AnyAutoEnzyme{M,Nothing}) where {M}
    return f
end

function get_f_and_df(f, ::AnyAutoEnzyme{M,<:Const}) where {M}
    return Const(f)
end

function get_f_and_df(f, ::AnyAutoEnzyme{M,<:Duplicated}) where {M}
    return Duplicated(f, make_zero(f))
end

get_annotation(::A) where {A<:Enzyme.Annotation} = A
get_annotation(::F) where {F} = Const{F}
