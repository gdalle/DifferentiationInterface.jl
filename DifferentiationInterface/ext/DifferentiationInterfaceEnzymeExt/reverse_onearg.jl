function seeded_autodiff_thunk(
    rmode::ReverseModeSplit{ReturnPrimal},
    dresult,
    f::FA,
    ::Type{RA},
    args::Vararg{Annotation,N},
) where {ReturnPrimal,FA<:Annotation,RA<:Annotation,N}
    forward, reverse = autodiff_thunk(rmode, FA, RA, typeof.(args)...)
    tape, result, shadow_result = forward(f, args...)
    if RA <: Active
        dresult_righttype = convert(typeof(result), dresult)
        dinputs = only(reverse(f, args..., dresult_righttype, tape))
    else
        shadow_result .+= dresult  # TODO: generalize beyond arrays
        dinputs = only(reverse(f, args..., tape))
    end
    if ReturnPrimal
        return (dinputs, result)
    else
        return (dinputs,)
    end
end

## Pullback

function DI.prepare_pullback(
    f::F,
    ::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return NoPullbackExtras()
end

### Out-of-place

function DI.value_and_pullback(
    f::F,
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    ys_and_dxs = map(ty.d) do dy
        y, tx = DI.value_and_pullback(f, extras, backend, x, Tangents(dy), contexts...)
        y, only(tx)
    end
    y = first(ys_and_dxs[1])
    dxs = last.(ys_and_dxs)
    tx = Tangents(dxs...)
    return y, tx
end

function DI.value_and_pullback(
    f::F,
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::Tangents{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = reverse_mode_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : Duplicated
    dinputs, result = seeded_autodiff_thunk(
        mode, only(ty), f_and_df, RA, Active(x), map(translate, contexts)...
    )
    return result, Tangents(first(dinputs))
end

function DI.value_and_pullback(
    f::F,
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = reverse_mode_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : Duplicated
    dx = make_zero(x)
    _, result = seeded_autodiff_thunk(
        mode, only(ty), f_and_df, RA, Duplicated(x, dx), map(translate, contexts)...
    )
    return result, Tangents(dx)
end

function DI.pullback(
    f::F,
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::Tangents{1},
    contexts::Vararg{Context,C},
) where {F,C}
    return last(DI.value_and_pullback(f, extras, backend, x, ty, contexts...))
end

### In-place

function DI.value_and_pullback!(
    f::F,
    tx::Tangents,
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    ys = map(tx.d, ty.d) do dx, dy
        y, _ = DI.value_and_pullback!(
            f, Tangents(dx), extras, backend, x, Tangents(dy), contexts...
        )
        y
    end
    y = first(ys)
    return y, tx
end

function DI.pullback!(
    f::F,
    tx::Tangents,
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    for b in eachindex(tx.d, ty.d)
        DI.pullback!(
            f, Tangents(tx.d[b]), extras, backend, x, Tangents(ty.d[b]), contexts...
        )
    end
    return tx
end

function DI.value_and_pullback!(
    f::F,
    tx::Tangents{1},
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = reverse_mode_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : Duplicated
    dx_righttype = convert(typeof(x), only(tx))
    make_zero!(dx_righttype)
    _, result = seeded_autodiff_thunk(
        mode,
        only(ty),
        f_and_df,
        RA,
        Duplicated(x, dx_righttype),
        map(translate, contexts)...,
    )
    only(tx) === dx_righttype || copyto!(only(tx), dx_righttype)
    return result, tx
end

function DI.pullback!(
    f::F,
    tx::Tangents{1},
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1},
    contexts::Vararg{Context,C},
) where {F,C}
    return last(DI.value_and_pullback!(f, tx, extras, backend, x, ty, contexts...))
end

## Gradient

function DI.prepare_gradient(
    f::F,
    ::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return NoGradientExtras()
end

function DI.gradient(
    f::F,
    ::NoGradientExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    derivs = gradient(
        reverse_mode_noprimal(backend), f_and_df, x, map(translate, contexts)...
    )
    return first(derivs)
end

function DI.gradient!(
    f::F,
    grad,
    ::NoGradientExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    dx_righttype = convert(typeof(x), grad)
    make_zero!(dx_righttype)
    autodiff(
        reverse_mode_noprimal(backend),
        f,
        Active,
        Duplicated(x, dx_righttype),
        map(translate, contexts)...,
    )
    dx_righttype === grad || copyto!(grad, dx_righttype)
    return grad
end

function DI.value_and_gradient(
    f::F,
    ::NoGradientExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    (; derivs, val) = gradient(
        reverse_mode_withprimal(backend), f_and_df, x, map(translate, contexts)...
    )
    return val, first(derivs)
end

function DI.value_and_gradient!(
    f::F,
    grad,
    ::NoGradientExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    dx_righttype = convert(typeof(x), grad)
    make_zero!(dx_righttype)
    _, y = autodiff(
        reverse_mode_withprimal(backend),
        f,
        Active,
        Duplicated(x, dx_righttype),
        map(translate, contexts)...,
    )
    dx_righttype === grad || copyto!(grad, dx_righttype)
    return y, grad
end

## Jacobian

struct EnzymeReverseOneArgJacobianExtras{M,B} <: JacobianExtras end

function DI.prepare_jacobian(f::F, backend::AutoEnzyme{<:ReverseMode,Nothing}, x) where {F}
    y = f(x)
    M = length(y)
    B = pick_batchsize(backend, M)
    return EnzymeReverseOneArgJacobianExtras{M,B}()
end

function DI.jacobian(
    f::F,
    ::EnzymeReverseOneArgJacobianExtras{M,B},
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {F,M,B}
    derivs = jacobian(reverse_mode_noprimal(backend), f, x; n_outs=Val((M,)), chunk=Val(B))
    jac_tensor = only(derivs)
    return maybe_reshape(jac_tensor, M, length(x))
end

function DI.value_and_jacobian(
    f::F,
    extras::EnzymeReverseOneArgJacobianExtras{M,B},
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {F,M,B}
    (; derivs, val) = jacobian(
        reverse_mode_withprimal(backend), f, x; n_outs=Val((M,)), chunk=Val(B)
    )
    jac_tensor = derivs
    return val, maybe_reshape(jac_tensor, M, length(x))
end

function DI.jacobian!(
    f::F,
    jac,
    extras::EnzymeReverseOneArgJacobianExtras,
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {F}
    return copyto!(jac, DI.jacobian(f, extras, backend, x))
end

function DI.value_and_jacobian!(
    f::F,
    jac,
    extras::EnzymeReverseOneArgJacobianExtras,
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {F}
    y, new_jac = DI.value_and_jacobian(f, extras, backend, x)
    return y, copyto!(jac, new_jac)
end
