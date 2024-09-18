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

mode_noprimal(::Type{ForwardMode{wp,FFIABI,A,B}}) where {wp,FFIABI,A,B} = ForwardMode{false,FFIABI,A,B}()
mode_noprimal(::Type{ReverseMode{wp,C,FFIABI,A,B}}) where {wp,C,FFIABI,A,B} = ReverseMode{false,C,FFIABI,A,B}()

mode_noprimal(mode::Mode) = mode_noprimal(typeof(mode))

mode_withprimal(::Type{ForwardMode{wp,FFIABI,A,B}}) where {wp,FFIABI,A,B} = ForwardMode{true,FFIABI,A,B}()
mode_withprimal(::Type{ReverseMode{wp,C,FFIABI,A,B}}) where {wp,C,FFIABI,A,B} = ReverseMode{true,FFIABI,A,B}()

mode_withprimal(mode::Mode) = mode_withprimal(typeof(mode))

mode_noprimal(backend::AnyAutoEnzyme) = mode_noprimal(backend.mode)
mode_noprimal(::AnyAutoEnzyme{Nothing}) = Forward

mode_withprimal(backend::AnyAutoEnzyme) = mode_withprimal(backend.mode)
mode_withprimal(::AnyAutoEnzyme{Nothing}) = Forward

DI.check_available(::AutoEnzyme) = true

# until https://github.com/EnzymeAD/Enzyme.jl/pull/1545 is merged
DI.pick_batchsize(::AnyAutoEnzyme, dimension::Integer) = min(dimension, 16)

function get_f_and_df(f, ::AnyAutoEnzyme{M,Nothing}) where {M}
    return f
end

function get_f_and_df(f, ::AnyAutoEnzyme{M,<:Const}) where {M}
    return Const(f)
end

function get_f_and_df(f, ::AnyAutoEnzyme{M,<:Duplicated}) where {M}
    return Duplicated(f, make_zero(f))
end

force_annotation(f::Annotation) = f
force_annotation(f) = Const(f)

# TODO: move this to EnzymeCore

function my_set_err_if_func_written(
    ::EnzymeCore.ReverseModeSplit{
        ReturnPrimal,A,ReturnShadow,Width,ModifiedBetween,ABI,ErrIfFuncWritten
    },
) where {ReturnPrimal,A,ReturnShadow,Width,ModifiedBetween,ABI,ErrIfFuncWritten}
    return EnzymeCore.ReverseModeSplit{
        ReturnPrimal,A,ReturnShadow,Width,ModifiedBetween,ABI,true
    }()
end
