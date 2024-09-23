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

function DI.prepare_pullback(f, ::AutoEnzyme{<:Union{ReverseMode,Nothing}}, x, ty::Tangents)
    return NoPullbackExtras()
end

### Out-of-place

function DI.value_and_pullback(
    f,
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
)
    tx = map(ty) do dy
        only(DI.pullback(f, extras, backend, x, Tangents(dy)))
    end
    y = f(x)  # TODO: optimize
    return y, tx
end

function DI.value_and_pullback(
    f,
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::Tangents{1},
)
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = reverse_mode_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : Duplicated
    dinputs, result = seeded_autodiff_thunk(mode, only(ty), f_and_df, RA, Active(x))
    return result, Tangents(only(dinputs))
end

function DI.value_and_pullback(
    f,
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1},
)
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = reverse_mode_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : Duplicated
    dx = make_zero(x)
    _, result = seeded_autodiff_thunk(mode, only(ty), f_and_df, RA, Duplicated(x, dx))
    return result, Tangents(dx)
end

function DI.pullback(
    f,
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::Tangents{1},
)
    return last(DI.value_and_pullback(f, extras, backend, x, ty))
end

### In-place

function DI.value_and_pullback!(
    f,
    tx::Tangents,
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
)
    for b in eachindex(tx.d, ty.d)
        DI.pullback!(f, Tangents(tx.d[b]), extras, backend, x, Tangents(ty.d[b]))
    end
    y = f(x)  # TODO: optimize
    return y, tx
end

function DI.pullback!(
    f,
    tx::Tangents,
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
)
    for b in eachindex(tx.d, ty.d)
        DI.pullback!(f, Tangents(tx.d[b]), extras, backend, x, Tangents(ty.d[b]))
    end
    return tx
end

function DI.value_and_pullback!(
    f,
    tx::Tangents{1},
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1},
)
    f_and_df = force_annotation(get_f_and_df(f, backend))
    mode = reverse_mode_split_withprimal(backend)
    RA = eltype(ty) <: Number ? Active : Duplicated
    dx_righttype = convert(typeof(x), only(tx))
    make_zero!(dx_righttype)
    _, result = seeded_autodiff_thunk(
        mode, only(ty), f_and_df, RA, Duplicated(x, dx_righttype)
    )
    only(tx) === dx_righttype || copyto!(only(tx), dx_righttype)
    return result, tx
end

function DI.pullback!(
    f,
    tx::Tangents{1},
    extras::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1},
)
    return last(DI.value_and_pullback!(f, tx, extras, backend, x, ty))
end

## Gradient

function DI.prepare_gradient(
    f, ::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}}, x
)
    return NoGradientExtras()
end

function DI.gradient(
    f,
    ::NoGradientExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    f_and_df = get_f_and_df(f, backend)
    derivs = gradient(reverse_mode_noprimal(backend), f_and_df, x)
    return only(derivs)
end

function DI.gradient!(
    f,
    grad,
    extras::NoGradientExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    return copyto!(grad, DI.gradient(f, extras, backend, x))
end

function DI.value_and_gradient(
    f,
    ::NoGradientExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    f_and_df = get_f_and_df(f, backend)
    (; derivs, val) = gradient(reverse_mode_withprimal(backend), f_and_df, x)
    return val, only(derivs)
end

function DI.value_and_gradient!(
    f,
    grad,
    ::NoGradientExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    dx_righttype = convert(typeof(x), grad)
    make_zero!(dx_righttype)
    _, y = autodiff(
        reverse_mode_withprimal(backend), f, Active, Duplicated(x, dx_righttype)
    )
    dx_righttype === grad || copyto!(grad, dx_righttype)
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
    derivs = jacobian(reverse_mode_noprimal(backend), f, x; n_outs=Val((M,)), chunk=Val(B))
    jac_tensor = only(derivs)
    return maybe_reshape(jac_tensor, M, length(x))
end

function DI.value_and_jacobian(
    f,
    extras::EnzymeReverseOneArgJacobianExtras{M,B},
    backend::AutoEnzyme{<:ReverseMode,Nothing},
    x,
) where {M,B}
    (; derivs, val) = jacobian(
        reverse_mode_withprimal(backend), f, x; n_outs=Val((M,)), chunk=Val(B)
    )
    jac_tensor = derivs
    return val, maybe_reshape(jac_tensor, M, length(x))
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
