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
    dresults::NTuple{B},
    f::FA,
    ::Type{RA},
    args::Vararg{Annotation,N},
) where {ReturnPrimal,B,FA<:Annotation,RA<:Annotation,N}
    rmode_rightwidth = ReverseSplitWidth(rmode, Val(B))
    forward, reverse = autodiff_thunk(rmode_rightwidth, FA, RA, typeof.(args)...)
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
    ty::NTuple,
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
    ty::NTuple{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = reverse_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : Duplicated
    dinputs, result = seeded_autodiff_thunk(
        mode, only(ty), f_and_df, RA, Active(x), map(translate, contexts)...
    )
    return result, (first(dinputs),)
end

function DI.value_and_pullback(
    f::F,
    ::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f_and_df = force_annotation(get_f_and_df(f, backend, Val(B)))
    mode = reverse_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : BatchDuplicated
    dinputs, result = batch_seeded_autodiff_thunk(
        mode, ty, f_and_df, RA, Active(x), map(translate, contexts)...
    )
    return result, values(first(dinputs))
end

function DI.value_and_pullback(
    f::F,
    ::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = reverse_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : Duplicated
    dx = make_zero(x)
    _, result = seeded_autodiff_thunk(
        mode, only(ty), f_and_df, RA, Duplicated(x, dx), map(translate, contexts)...
    )
    return result, (dx,)
end

function DI.value_and_pullback(
    f::F,
    ::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f_and_df = force_annotation(get_f_and_df(f, backend, Val(B)))
    mode = reverse_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : BatchDuplicated
    tx = ntuple(_ -> make_zero(x), Val(B))
    _, result = batch_seeded_autodiff_thunk(
        mode, ty, f_and_df, RA, BatchDuplicated(x, tx), map(translate, contexts)...
    )
    return result, tx
end

function DI.pullback(
    f::F,
    prep::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return last(DI.value_and_pullback(f, prep, backend, x, ty, contexts...))
end

### In-place

function DI.value_and_pullback!(
    f::F,
    tx::NTuple{1},
    ::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = reverse_split_withprimal(backend)
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
    tx::NTuple{B},
    ::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f_and_df = force_annotation(get_f_and_df(f, backend, Val(B)))
    mode = reverse_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : BatchDuplicated
    tx_righttype = map(Fix1(convert, typeof(x)), tx)
    make_zero!(tx_righttype)
    _, result = batch_seeded_autodiff_thunk(
        mode,
        ty,
        f_and_df,
        RA,
        BatchDuplicated(x, tx_righttype),
        map(translate, contexts)...,
    )
    foreach(copyto!, tx, tx_righttype)
    return result, tx
end

function DI.pullback!(
    f::F,
    tx::NTuple,
    prep::NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return last(DI.value_and_pullback!(f, tx, prep, backend, x, ty, contexts...))
end

## Gradient

### Without preparation

function DI.gradient(
    f::F,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    ders = gradient(reverse_noprimal(backend), f_and_df, x, map(translate, contexts)...)
    grad = first(ders)
    return grad
end

function DI.value_and_gradient(
    f::F,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    ders, y = gradient(
        reverse_withprimal(backend), f_and_df, x, map(translate, contexts)...
    )
    grad = first(ders)
    return y, grad
end

### With preparation

struct EnzymeGradientPrep{G} <: GradientPrep
    grad_righttype::G
end

function DI.prepare_gradient(
    f::F,
    ::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    grad_righttype = make_zero(x)
    return EnzymeGradientPrep(grad_righttype)
end

function DI.gradient(
    f::F,
    ::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    ders = gradient(reverse_noprimal(backend), f_and_df, x, map(translate, contexts)...)
    grad = first(ders)
    return grad
end

function DI.gradient!(
    f::F,
    grad,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    grad_righttype = grad isa typeof(x) ? grad : prep.grad_righttype
    make_zero!(grad_righttype)
    autodiff(
        reverse_noprimal(backend),
        f_and_df,
        Active,
        Duplicated(x, grad_righttype),
        map(translate, contexts)...,
    )
    grad === grad_righttype || copyto!(grad, grad_righttype)
    return grad
end

function DI.value_and_gradient(
    f::F,
    ::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    ders, y = gradient(
        reverse_withprimal(backend), f_and_df, x, map(translate, contexts)...
    )
    grad = first(ders)
    return y, grad
end

function DI.value_and_gradient!(
    f::F,
    grad,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    grad_righttype = grad isa typeof(x) ? grad : prep.grad_righttype
    make_zero!(grad_righttype)
    _, y = autodiff(
        reverse_withprimal(backend),
        f_and_df,
        Active,
        Duplicated(x, grad_righttype),
        map(translate, contexts)...,
    )
    grad === grad_righttype || copyto!(grad, grad_righttype)
    return y, grad
end

## Jacobian

struct EnzymeReverseOneArgJacobianPrep{Sy,B} <: JacobianPrep end

function EnzymeReverseOneArgJacobianPrep(::Val{Sy}, ::Val{B}) where {Sy,B}
    return EnzymeReverseOneArgJacobianPrep{Sy,B}()
end

function DI.prepare_jacobian(f::F, backend::AutoEnzyme{<:ReverseMode,Nothing}, x) where {F}
    y = f(x)
    Sy = size(y)
    valB = pick_batchsize(backend, prod(Sy))
    return EnzymeReverseOneArgJacobianPrep(Val(Sy), valB)
end

function DI.jacobian(
    f::F,
    ::EnzymeReverseOneArgJacobianPrep{Sy,B},
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {F,Sy,B}
    derivs = jacobian(reverse_noprimal(backend), f, x; n_outs=Val(Sy), chunk=Val(B))
    jac_tensor = only(derivs)
    return maybe_reshape(jac_tensor, prod(Sy), length(x))
end

function DI.value_and_jacobian(
    f::F,
    ::EnzymeReverseOneArgJacobianPrep{Sy,B},
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {F,Sy,B}
    (; derivs, val) = jacobian(
        reverse_withprimal(backend), f, x; n_outs=Val(Sy), chunk=Val(B)
    )
    jac_tensor = only(derivs)
    return val, maybe_reshape(jac_tensor, prod(Sy), length(x))
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
