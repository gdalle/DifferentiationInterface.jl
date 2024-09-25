# until https://github.com/EnzymeAD/Enzyme.jl/pull/1545 is merged
DI.pick_batchsize(::AutoEnzyme, dimension::Integer) = min(dimension, 16)

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
            EnzymeCore.DuplicatedNoNeed,
            BatchDuplicated,
            EnzymeCore.BatchDuplicatedFunc,
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

function mode_noprimal(
    ::Type{ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}}
) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}
    return ForwardMode{false,ABI,ErrIfFuncWritten,RuntimeActivity}()
end

function mode_withprimal(
    ::Type{ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}}
) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}
    return ForwardMode{true,ABI,ErrIfFuncWritten,RuntimeActivity}()
end

function mode_noprimal(
    ::Type{ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}}
) where {ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}
    return ReverseMode{false,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}()
end

function mode_withprimal(
    ::Type{ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}}
) where {ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}
    return ReverseMode{true,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}()
end

function mode_noprimal(
    ::Type{
        ReverseModeSplit{
            ReturnPrimal,
            ReturnShadow,
            RuntimeActivity,
            Width,
            ModifiedBetween,
            ABI,
            ErrIfFuncWritten,
        },
    },
) where {
    ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI,ErrIfFuncWritten
}
    return ReverseModeSplit{
        false,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI,ErrIfFuncWritten
    }()
end

function mode_withprimal(
    ::Type{
        ReverseModeSplit{
            ReturnPrimal,
            ReturnShadow,
            RuntimeActivity,
            Width,
            ModifiedBetween,
            ABI,
            ErrIfFuncWritten,
        },
    },
) where {
    ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI,ErrIfFuncWritten
}
    return ReverseModeSplit{
        true,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI,ErrIfFuncWritten
    }()
end

function mode_split(
    ::Type{ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}}
) where {ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}
    ReturnShadow = true
    Width = 0
    ModifiedBetween = true
    return ReverseModeSplit{
        ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI,ErrIfFuncWritten
    }()
end

function set_width(
    ::ReverseModeSplit{
        ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI,ErrIfFuncWritten
    },
    ::Val{NewWidth},
) where {
    ReturnPrimal,
    ReturnShadow,
    RuntimeActivity,
    Width,
    ModifiedBetween,
    ABI,
    ErrIfFuncWritten,
    NewWidth,
}
    return ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        RuntimeActivity,
        NewWidth,
        ModifiedBetween,
        ABI,
        ErrIfFuncWritten,
    }()
end

mode_noprimal(mode::Mode) = mode_noprimal(typeof(mode))
mode_withprimal(mode::Mode) = mode_withprimal(typeof(mode))
mode_split(mode::Mode) = mode_split(typeof(mode))

forward_mode_noprimal(backend::AutoEnzyme{<:ForwardMode}) = mode_noprimal(backend.mode)
forward_mode_noprimal(::AutoEnzyme{Nothing}) = Forward

forward_mode_withprimal(backend::AutoEnzyme{<:ForwardMode}) = mode_withprimal(backend.mode)
forward_mode_withprimal(::AutoEnzyme{Nothing}) = ForwardWithPrimal

reverse_mode_noprimal(backend::AutoEnzyme{<:ReverseMode}) = mode_noprimal(backend.mode)
reverse_mode_noprimal(::AutoEnzyme{Nothing}) = Reverse

reverse_mode_withprimal(backend::AutoEnzyme{<:ReverseMode}) = mode_withprimal(backend.mode)
reverse_mode_withprimal(::AutoEnzyme{Nothing}) = ReverseWithPrimal

set_err(mode::Mode, ::AutoEnzyme{<:Any,Nothing}) = EnzymeCore.set_err_if_func_written(mode)
set_err(mode::Mode, ::AutoEnzyme{<:Any,<:Annotation}) = mode

function reverse_mode_split_noprimal(backend::AutoEnzyme)
    return set_err(mode_split(reverse_mode_noprimal(backend)), backend)
end

function reverse_mode_split_withprimal(backend::AutoEnzyme)
    return set_err(mode_split(reverse_mode_withprimal(backend)), backend)
end

function maybe_reshape(A::AbstractMatrix, m, n)
    @assert size(A) == (m, n)
    return A
end

function maybe_reshape(A::AbstractArray, m, n)
    return reshape(A, m, n)
end
