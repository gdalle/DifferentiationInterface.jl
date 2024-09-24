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

function batch_seeded_autodiff_thunk(
    rmode::ReverseModeSplit{ReturnPrimal},
    dresults::NTuple,
    f::FA,
    ::Type{RA},
    args::Vararg{Annotation,N},
) where {ReturnPrimal,FA<:Annotation,RA<:Annotation,N}
    forward, reverse = autodiff_thunk(rmode, FA, RA, typeof.(args)...)
    tape, result, shadow_results = forward(f, args...)
    if RA <: Active
        dresults_righttype = map(Fix1(convert, typeof(result)), dresults)
        dinputs = only(reverse(f, args..., dresults_righttype, tape))
    else
        foreach(shadow_results, dresults) do d0, d
            d0 .+= d  # use recursive_add here?
        end
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
    return NoPullbackPrep()
end

### Out-of-place

function DI.value_and_pullback(
    f::F,
    ::NoPullbackPrep,
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
    prep::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    # TODO: improve
    ys_and_dxs = map(ty.d) do dy
        y, tx = DI.value_and_pullback(f, prep, backend, x, Tangents(dy), contexts...)
        y, only(tx)
    end
    y = first(ys_and_dxs[1])
    dxs = last.(ys_and_dxs)
    return y, Tangents(dxs...)
end

function DI.value_and_pullback(
    f::F,
    ::NoPullbackPrep,
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

function DI.value_and_pullback(
    f::F,
    ::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f_and_df = force_annotation(get_f_and_df(f, backend, Val(B)))
    mode = reverse_mode_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : BatchDuplicated
    dxs = ntuple(_ -> make_zero(x), Val(B))
    _, result = batch_seeded_autodiff_thunk(
        mode, NTuple(ty), f_and_df, RA, BatchDuplicated(x, dxs), map(translate, contexts)...
    )
    return result, Tangents(dxs...)
end

function DI.pullback(
    f::F,
    prep::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return last(DI.value_and_pullback(f, prep, backend, x, ty, contexts...))
end

### In-place

function DI.value_and_pullback!(
    f::F,
    tx::Tangents{1},
    ::NoPullbackPrep,
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

function DI.value_and_pullback!(
    f::F,
    tx::Tangents{B},
    ::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f_and_df = force_annotation(get_f_and_df(f, backend, Val(B)))
    mode = reverse_mode_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : BatchDuplicated
    dxs_righttype = map(Fix1(convert, typeof(x)), NTuple(tx))
    make_zero!(dxs_righttype)
    _, result = batch_seeded_autodiff_thunk(
        mode,
        NTuple(ty),
        f_and_df,
        RA,
        BatchDuplicated(x, dxs_righttype),
        map(translate, contexts)...,
    )
    foreach(copyto!, NTuple(tx), dxs_righttype)
    return result, tx
end

function DI.pullback!(
    f::F,
    tx::Tangents,
    prep::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return last(DI.value_and_pullback!(f, tx, prep, backend, x, ty, contexts...))
end

## Gradient

function DI.prepare_gradient(
    f::F,
    ::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    return NoGradientPrep()
end

function DI.gradient(
    f::F,
    ::NoGradientPrep,
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
    ::NoGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    dx_righttype = convert(typeof(x), grad)
    make_zero!(dx_righttype)
    autodiff(
        reverse_mode_noprimal(backend),
        f_and_df,
        Active,
        Duplicated(x, dx_righttype),
        map(translate, contexts)...,
    )
    dx_righttype === grad || copyto!(grad, dx_righttype)
    return grad
end

function DI.value_and_gradient(
    f::F,
    ::NoGradientPrep,
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
    ::NoGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    dx_righttype = convert(typeof(x), grad)
    make_zero!(dx_righttype)
    _, y = autodiff(
        reverse_mode_withprimal(backend),
        f_and_df,
        Active,
        Duplicated(x, dx_righttype),
        map(translate, contexts)...,
    )
    dx_righttype === grad || copyto!(grad, dx_righttype)
    return y, grad
end

## Jacobian

struct EnzymeReverseOneArgJacobianPrep{M,B} <: JacobianPrep end

function DI.prepare_jacobian(f::F, backend::AutoEnzyme{<:ReverseMode,Nothing}, x) where {F}
    y = f(x)
    M = length(y)
    B = pick_batchsize(backend, M)
    return EnzymeReverseOneArgJacobianPrep{M,B}()
end

function DI.jacobian(
    f::F,
    ::EnzymeReverseOneArgJacobianPrep{M,B},
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {F,M,B}
    derivs = jacobian(reverse_mode_noprimal(backend), f, x; n_outs=Val((M,)), chunk=Val(B))
    jac_tensor = only(derivs)
    return maybe_reshape(jac_tensor, M, length(x))
end

function DI.value_and_jacobian(
    f::F,
    prep::EnzymeReverseOneArgJacobianPrep{M,B},
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
    prep::EnzymeReverseOneArgJacobianPrep,
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {F}
    return copyto!(jac, DI.jacobian(f, prep, backend, x))
end

function DI.value_and_jacobian!(
    f::F,
    jac,
    prep::EnzymeReverseOneArgJacobianPrep,
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {F}
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x)
    return y, copyto!(jac, new_jac)
end
