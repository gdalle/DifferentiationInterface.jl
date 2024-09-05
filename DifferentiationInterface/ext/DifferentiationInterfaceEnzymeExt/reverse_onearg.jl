## Pullback

function DI.prepare_pullback(
    f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}}, x, ty::Tangents
)
    return NoPullbackExtras()
end

function DI.value_and_pullback(
    f,
    extras::NoPullbackExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
)
    dxs = map(ty.d) do dy
        only(DI.pullback(f, extras, backend, x, SingleTangent(dy)))
    end
    y = f(x)
    return y, Tangents(dxs)
end

### Out-of-place

function DI.value_and_pullback(
    f,
    ::NoPullbackExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},function_annotation},
    x::Number,
    ty::Tangents{1},
) where {function_annotation}
    if eltype(ty) <: Number
        dy = only(ty)
        f_and_df = get_f_and_df(f, backend)
        der, y = if backend isa AutoDeferredEnzyme
            autodiff_deferred(ReverseWithPrimal, f_and_df, Active, Active(x))
        else
            autodiff(ReverseWithPrimal, f_and_df, Active, Active(x))
        end
        new_dx = dy * only(der)
        return y, SingleTangent(new_dx)
    else
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
end

function DI.value_and_pullback(
    f,
    extras::NoPullbackExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},function_annotation},
    x,
    ty::Tangents{1},
) where {function_annotation}
    if eltype(ty) <: Number
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
    else
        dx = make_zero(x)
        return DI.value_and_pullback!(f, SingleTangent(dx), extras, backend, x, ty)
    end
end

function DI.pullback(
    f,
    extras::NoPullbackExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
)
    return DI.value_and_pullback(f, extras, backend, x, ty)[2]
end

### In-place

function DI.value_and_pullback!(
    f,
    tx::Tangents{1},
    ::NoPullbackExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},function_annotation},
    x,
    ty::Tangents{1},
) where {function_annotation}
    if eltype(ty) <: Number
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
    else
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
end

function DI.pullback!(
    f,
    tx::Tangents,
    extras::NoPullbackExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
)
    return DI.value_and_pullback!(f, tx, extras, backend, x, ty)[2]
end

## Gradient

function DI.prepare_gradient(
    f, ::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}}, x
)
    return NoGradientExtras()
end

function DI.gradient(
    f,
    ::NoGradientExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
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
    ::NoGradientExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
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
    ::NoGradientExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    return DI.value_and_pullback(f, NoPullbackExtras(), backend, x, true)
end

function DI.value_and_gradient!(
    f,
    grad,
    ::NoGradientExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    return DI.value_and_pullback!(f, grad, NoPullbackExtras(), backend, x, true)
end

## Jacobian

struct EnzymeReverseOneArgJacobianExtras{M,B} <: JacobianExtras end

function DI.prepare_jacobian(f, backend::AutoEnzyme{<:ReverseMode,Nothing}, x)
    y = f(x)
    M = length(y)
    B = pick_batchsize(backend, M)
    return EnzymeReverseOneArgJacobianExtras{M,B}()
end

function DI.jacobian(
    f,
    ::EnzymeReverseOneArgJacobianExtras{M,B},
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {M,B}
    jac_wrongshape = jacobian(reverse_mode(backend), f, x, Val(M), Val(B))
    nx = length(x)
    ny = length(jac_wrongshape) ÷ length(x)
    return reshape(jac_wrongshape, ny, nx)
end

function DI.value_and_jacobian(
    f,
    extras::EnzymeReverseOneArgJacobianExtras,
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
)
    return f(x), DI.jacobian(f, extras, backend, x)
end

function DI.jacobian!(
    f,
    jac,
    extras::EnzymeReverseOneArgJacobianExtras,
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
)
    return copyto!(jac, DI.jacobian(f, extras, backend, x))
end

function DI.value_and_jacobian!(
    f,
    jac,
    extras::EnzymeReverseOneArgJacobianExtras,
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
)
    y, new_jac = DI.value_and_jacobian(f, extras, backend, x)
    return y, copyto!(jac, new_jac)
end
