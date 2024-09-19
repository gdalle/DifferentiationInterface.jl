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
    tx = map(ty) do dy
        only(DI.pullback(f, extras, backend, x, Tangents(dy)))
    end
    y = f(x)
    return y, tx
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
        return y, Tangents(new_dx)
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
        return y, Tangents(new_dx)
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
        return y, Tangents(dx_sametype)
    else
        dx = make_zero(x)
        return DI.value_and_pullback!(f, Tangents(dx), extras, backend, x, ty)
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
    gradient(mode_noprimal(backend), f_and_df, x)[1]
end

function DI.gradient!(
    f,
    grad,
    extras::NoGradientExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    return copyto!(grad, DI.gradient(f, extras, backend, x))
end

function DI.value_and_gradient(
    f,
    ::NoGradientExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    f_and_df = get_f_and_df(f, backend)
    gr = gradient(mode_withprimal(backend), f_and_df, x)
    return gr.val, gr.derivs[1]    
end

function DI.value_and_gradient!(
    f,
    grad,
    ::NoGradientExtras,
    backend::AnyAutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    y, _ = DI.value_and_pullback!(
        f, Tangents(grad), NoPullbackExtras(), backend, x, Tangents(true)
    )
    return y, grad
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
    J = jacobian(mode_noprimal(backend), f, x; n_outs=Val((M,)), chunk=Val(B))[1]
    flatjac(x, J)
end

function DI.value_and_jacobian(
    f,
    extras::EnzymeReverseOneArgJacobianExtras{M,B},
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {M,B}
    jac = jacobian(mode_withprimal(backend), f, x; n_outs=Val((M,)), chunk=Val(B))
    return jac.val, flatjac(x, jac.derivs[1])
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
