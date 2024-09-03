## DI boilerplate

DI.check_available(::AutoEnzyme) = true

# until https://github.com/EnzymeAD/Enzyme.jl/pull/1545 is merged
DI.pick_batchsize(::AnyAutoEnzyme, dimension::Integer) = min(dimension, 16)

## Useful closures

struct Converter{X} end
Converter(::X) where {X} = Converter{X}()
(c::Converter{X})(y) where {X} = convert(X, y)

struct Zero{X}
    x::X
end
(z::Zero{X})(_) where {X} = make_zero(z.x)

## Nested backend

struct AutoDeferredEnzyme{M,A} <: ADTypes.AbstractADType
    mode::M
end

function ADTypes.mode(backend::AutoDeferredEnzyme{M,A}) where {M,A}
    return ADTypes.mode(AutoEnzyme{M,A}(backend.mode))
end

function DI.nested(backend::AutoEnzyme{M,A}) where {M,A}
    return AutoDeferredEnzyme{M,A}(backend.mode)
end

const AnyAutoEnzyme{M,A} = Union{AutoEnzyme{M,A},AutoDeferredEnzyme{M,A}}

## Mode objects

# forward mode if possible
forward_mode(backend::AnyAutoEnzyme{<:Mode}) = backend.mode
forward_mode(::AnyAutoEnzyme{Nothing}) = Forward

# reverse mode if possible
reverse_mode(backend::AnyAutoEnzyme{<:Mode}) = backend.mode
reverse_mode(::AnyAutoEnzyme{Nothing}) = Reverse

## Annotations

function get_f_and_df(f, ::AnyAutoEnzyme{M,Nothing}) where {M}
    return f
end

function get_f_and_df(f, ::AnyAutoEnzyme{M,Nothing}, t::Tangents) where {M}
    return f
end

function get_f_and_df(f, ::AnyAutoEnzyme{M,<:Const}) where {M}
    return Const(f)
end

function get_f_and_df(f, ::AnyAutoEnzyme{M,<:Const}, t::Tangents) where {M}
    return Const(f)
end

function get_f_and_df(f, ::AnyAutoEnzyme{M,<:Union{Duplicated,BatchDuplicated}}) where {M}
    return Duplicated(f, make_zero(f))
end

function get_f_and_df(
    f, ::AnyAutoEnzyme{M,<:Union{Duplicated,BatchDuplicated}}, t::Tangents
) where {M}
    return BatchDuplicated(f, map(Zero(f), t.d))
end

force_annotation(f::Annotation) = f
force_annotation(f) = Const(f)

# TODO: move this to EnzymeCore

function my_set_err_if_func_written(
    ::EnzymeCore.ReverseModeSplit{
        ReturnPrimal,ReturnShadow,Width,ModifiedBetween,ABI,ErrIfFuncWritten
    },
) where {ReturnPrimal,ReturnShadow,Width,ModifiedBetween,ABI,ErrIfFuncWritten}
    return EnzymeCore.ReverseModeSplit{
        ReturnPrimal,ReturnShadow,Width,ModifiedBetween,ABI,true
    }()
end
