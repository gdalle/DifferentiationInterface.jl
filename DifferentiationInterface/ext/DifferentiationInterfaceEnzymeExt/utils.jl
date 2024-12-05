# until https://github.com/EnzymeAD/Enzyme.jl/pull/1545 is merged
function DI.BatchSizeSettings(::AutoEnzyme, N::Integer)
    B = DI.reasonable_batchsize(N, 16)
    return DI.BatchSizeSettings{B}(N)
end

to_val(::DI.BatchSizeSettings{B}) where {B} = Val(B)

## Annotations

function get_f_and_df(f::F, ::AutoEnzyme{M,Nothing}, ::Val{B}=Val(1)) where {F,M,B}
    return f
end

function get_f_and_df(f::F, ::AutoEnzyme{M,<:Const}, ::Val{B}=Val(1)) where {F,M,B}
    return Const(f)
end

function get_f_and_df(
    f::F,
    ::AutoEnzyme{
        M,
        <:Union{
            Duplicated,
            MixedDuplicated,
            BatchDuplicated,
            BatchMixedDuplicated,
            EnzymeCore.DuplicatedNoNeed,
            EnzymeCore.BatchDuplicatedNoNeed,
        },
    },
    ::Val{B}=Val(1),
) where {F,M,B}
    # TODO: needs more sophistication for mixed activities
    if B == 1
        return Duplicated(f, make_zero(f))
    else
        return BatchDuplicated(f, ntuple(_ -> make_zero(f), Val(B)))
    end
end

force_annotation(f::F) where {F<:Annotation} = f
force_annotation(f::F) where {F} = Const(f)

translate(c::DI.Constant) = Const(DI.unwrap(c))

## Modes

forward_noprimal(backend::AutoEnzyme{<:ForwardMode}) = NoPrimal(backend.mode)
forward_noprimal(::AutoEnzyme{Nothing}) = Forward

forward_withprimal(backend::AutoEnzyme{<:ForwardMode}) = WithPrimal(backend.mode)
forward_withprimal(::AutoEnzyme{Nothing}) = ForwardWithPrimal

reverse_noprimal(backend::AutoEnzyme{<:ReverseMode}) = NoPrimal(backend.mode)
reverse_noprimal(::AutoEnzyme{Nothing}) = Reverse

reverse_withprimal(backend::AutoEnzyme{<:ReverseMode}) = WithPrimal(backend.mode)
reverse_withprimal(::AutoEnzyme{Nothing}) = ReverseWithPrimal

function reverse_split_withprimal(backend::AutoEnzyme{<:ReverseMode})
    return set_err(WithPrimal(Split(backend.mode)), backend)
end

function reverse_split_withprimal(backend::AutoEnzyme{Nothing})
    return set_err(ReverseSplitWithPrimal, backend)
end

set_err(mode::Mode, ::AutoEnzyme{<:Any,Nothing}) = EnzymeCore.set_err_if_func_written(mode)
set_err(mode::Mode, ::AutoEnzyme{<:Any,<:Annotation}) = mode

function maybe_reshape(A::AbstractMatrix, m, n)
    @assert size(A) == (m, n)
    return A
end

function maybe_reshape(A::AbstractArray, m, n)
    return reshape(A, m, n)
end

annotate(::Type{Active{T}}, x, dx) where {T} = Active(x)
annotate(::Type{Duplicated{T}}, x, dx) where {T} = Duplicated(x, dx)

function annotate(::Type{BatchDuplicated{T,B}}, x, tx::NTuple{B}) where {T,B}
    return BatchDuplicated(x, tx)
end

batchify_activity(::Type{Active{T}}, ::Val{B}) where {T,B} = Active{T}
batchify_activity(::Type{Duplicated{T}}, ::Val{B}) where {T,B} = BatchDuplicated{T,B}
