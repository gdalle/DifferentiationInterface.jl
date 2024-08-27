## Pullback

function DI.prepare_pullback(
    f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, ty::Tangents
)
    return NoPullbackExtras()
end

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
    extras::NoPullbackExtras,
)
    dxs = map(ty.d) do dy
        DI.pullback(f, backend, x, dy, extras)
    end
    y = f(x)
    return y, Tangents(dxs)
end

### Out-of-place

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::Tangents{1,<:Number},
    ::NoPullbackExtras,
)
    dy = only(ty)
    f_and_df = get_f_and_df(f, backend)
    der, y = if backend isa AutoDeferredEnzyme
        autodiff_deferred(ReverseWithPrimal, f_and_df, Active, Active(x))
    else
        autodiff(ReverseWithPrimal, f_and_df, Active, Active(x))
    end
    new_dx = dy * only(der)
    return y, SingleTangent(new_dx)
end

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},function_annotation},
    x::Number,
    ty::Tangents{1},
    ::NoPullbackExtras,
) where {function_annotation}
    dy = only(ty)
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = if function_annotation <: Annotation
        ReverseSplitWithPrimal
    else
        my_set_err_if_func_written(ReverseSplitWithPrimal)
    end
    forw, rev = autodiff_thunk(mode, typeof(f_and_df), Duplicated, typeof(Active(x)))
    tape, y, new_dy = forw(f_and_df, Active(x))
    copyto!(new_dy, dy)
    new_dx = only(only(rev(f_and_df, Active(x), tape)))
    return y, SingleTangent(new_dx)
end

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1,<:Number},
    ::NoPullbackExtras,
)
    dy = only(ty)
    f_and_df = get_f_and_df(f, backend)
    dx_sametype = make_zero(x)
    x_and_dx = Duplicated(x, dx_sametype)
    _, y = if backend isa AutoDeferredEnzyme
        autodiff_deferred(ReverseWithPrimal, f_and_df, Active, x_and_dx)
    else
        autodiff(ReverseWithPrimal, f_and_df, Active, x_and_dx)
    end
    if !isone(dy)
        # TODO: generalize beyond Arrays?
        dx_sametype .*= dy
    end
    return y, SingleTangent(dx_sametype)
end

function DI.value_and_pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1},
    extras::NoPullbackExtras,
)
    dx = make_zero(x)
    return DI.value_and_pullback!(f, Tangents(dx), backend, x, ty, extras)
end

function DI.pullback(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
    extras::NoPullbackExtras,
)
    return DI.value_and_pullback(f, backend, x, ty, extras)[2]
end

### In-place

function DI.value_and_pullback!(
    f,
    tx::Tangents{1},
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1,<:Number},
    ::NoPullbackExtras,
)
    dx, dy = only(tx), only(ty)
    f_and_df = get_f_and_df(f, backend)
    dx_sametype = convert(typeof(x), dx)
    make_zero!(dx_sametype)
    x_and_dx = Duplicated(x, dx_sametype)
    _, y = if backend isa AutoDeferredEnzyme
        autodiff_deferred(ReverseWithPrimal, f_and_df, Active, x_and_dx)
    else
        autodiff(ReverseWithPrimal, f_and_df, Active, x_and_dx)
    end
    if !isone(dy)
        # TODO: generalize beyond Arrays?
        dx_sametype .*= dy
    end
    copyto!(dx, dx_sametype)
    return y, tx
end

function DI.value_and_pullback!(
    f,
    tx::Tangents{1},
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},function_annotation},
    x,
    ty::Tangents{1},
    ::NoPullbackExtras,
) where {function_annotation}
    dx, dy = only(tx), only(ty)
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = if function_annotation <: Annotation
        ReverseSplitWithPrimal
    else
        my_set_err_if_func_written(ReverseSplitWithPrimal)
    end
    dx_sametype = convert(typeof(x), dx)
    make_zero!(dx_sametype)
    x_and_dx = Duplicated(x, dx_sametype)
    forw, rev = autodiff_thunk(mode, typeof(f_and_df), Duplicated, typeof(x_and_dx))
    tape, y, new_dy = forw(f_and_df, x_and_dx)
    copyto!(new_dy, dy)
    rev(f_and_df, x_and_dx, tape)
    copyto!(dx, dx_sametype)
    return y, tx
end

function DI.pullback!(
    f,
    tx::Tangents,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
    extras::NoPullbackExtras,
)
    return DI.value_and_pullback!(f, tx, backend, x, ty, extras)[2]
end

## Gradient

function DI.prepare_gradient(
    f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}}, x
)
    return NoGradientExtras()
end

function DI.gradient(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    ::NoGradientExtras,
)
    f_and_df = get_f_and_df(f, backend)
    if backend isa AutoDeferredEnzyme
        grad = make_zero(x)
        autodiff_deferred(reverse_mode(backend), f_and_df, Active, Duplicated(x, grad))
        return grad
    else
        return gradient(reverse_mode(backend), f_and_df, x)
    end
end

function DI.gradient!(
    f,
    grad,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    ::NoGradientExtras,
)
    f_and_df = get_f_and_df(f, backend)
    grad_sametype = convert(typeof(x), grad)
    make_zero!(grad_sametype)
    if backend isa AutoDeferredEnzyme
        autodiff_deferred(
            reverse_mode(backend), f_and_df, Active, Duplicated(x, grad_sametype)
        )
    else
        gradient!(reverse_mode(backend), grad_sametype, f_and_df, x)
    end
    return copyto!(grad, grad_sametype)
end

function DI.value_and_gradient(
    f,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    ::NoGradientExtras,
)
    return DI.value_and_pullback(f, backend, x, true, NoPullbackExtras())
end

function DI.value_and_gradient!(
    f,
    grad,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    ::NoGradientExtras,
)
    return DI.value_and_pullback!(f, grad, backend, x, true, NoPullbackExtras())
end

## Jacobian

# see https://github.com/EnzymeAD/Enzyme.jl/issues/1391
