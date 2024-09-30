## Pushforward

function DI.prepare_pushforward(
    f::F,
    ::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    return NoPushforwardPrep()
end

function DI.value_and_pushforward(
    f::F,
    ::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    dx_sametype = convert(typeof(x), only(tx))
    x_and_dx = Duplicated(x, dx_sametype)
    dy, y = autodiff(
        forward_withprimal(backend), f_and_df, x_and_dx, map(translate, contexts)...
    )
    return y, (dy,)
end

function DI.value_and_pushforward(
    f::F,
    ::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f_and_df = get_f_and_df(f, backend, Val(B))
    tx_sametype = map(Fix1(convert, typeof(x)), tx)
    x_and_tx = BatchDuplicated(x, tx_sametype)
    ty, y = autodiff(
        forward_withprimal(backend), f_and_df, x_and_tx, map(translate, contexts)...
    )
    return y, values(ty)
end

function DI.pushforward(
    f::F,
    ::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    dx_sametype = convert(typeof(x), only(tx))
    x_and_dx = Duplicated(x, dx_sametype)
    dy = only(
        autodiff(forward_noprimal(backend), f_and_df, x_and_dx, map(translate, contexts)...)
    )
    return (dy,)
end

function DI.pushforward(
    f::F,
    ::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f_and_df = get_f_and_df(f, backend, Val(B))
    tx_sametype = map(Fix1(convert, typeof(x)), tx)
    x_and_tx = BatchDuplicated(x, tx_sametype)
    ty = only(
        autodiff(forward_noprimal(backend), f_and_df, x_and_tx, map(translate, contexts)...)
    )
    return values(ty)
end

function DI.value_and_pushforward!(
    f::F,
    ty::NTuple,
    prep::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    # dy cannot be passed anyway
    y, new_ty = DI.value_and_pushforward(f, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return y, ty
end

function DI.pushforward!(
    f::F,
    ty::NTuple,
    prep::NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{Context,C},
) where {F,C}
    # dy cannot be passed anyway
    new_ty = DI.pushforward(f, prep, backend, x, tx, contexts...)
    foreach(copyto!, ty, new_ty)
    return ty
end

## Gradient

struct EnzymeForwardGradientPrep{B,O} <: GradientPrep
    shadows::O
end

function DI.prepare_gradient(
    f::F, backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}}, x
) where {F}
    B = pick_batchsize(backend, length(x))
    shadows = create_shadows(Val(B), x)
    return EnzymeForwardGradientPrep{B,typeof(shadows)}(shadows)
end

function DI.gradient(
    f::F,
    prep::EnzymeForwardGradientPrep{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    f_and_df = get_f_and_df(f, backend)
    derivs = gradient(
        forward_noprimal(backend), f_and_df, x; chunk=Val(B), shadows=prep.shadows
    )
    return only(derivs)
end

function DI.value_and_gradient(
    f::F,
    prep::EnzymeForwardGradientPrep{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    f_and_df = get_f_and_df(f, backend)
    (; derivs, val) = gradient(
        forward_withprimal(backend), f_and_df, x; chunk=Val(B), shadows=prep.shadows
    )
    return val, only(derivs)
end

function DI.gradient!(
    f::F,
    grad,
    prep::EnzymeForwardGradientPrep{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    return copyto!(grad, DI.gradient(f, prep, backend, x))
end

function DI.value_and_gradient!(
    f::F,
    grad,
    prep::EnzymeForwardGradientPrep{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    y, new_grad = DI.value_and_gradient(f, prep, backend, x)
    return y, copyto!(grad, new_grad)
end

## Jacobian

struct EnzymeForwardOneArgJacobianPrep{B,O} <: JacobianPrep
    shadows::O
    output_length::Int
end

function DI.prepare_jacobian(
    f::F, backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}}, x
) where {F}
    y = f(x)
    B = pick_batchsize(backend, length(x))
    shadows = create_shadows(Val(B), x)
    return EnzymeForwardOneArgJacobianPrep{B,typeof(shadows)}(shadows, length(y))
end

function DI.jacobian(
    f::F,
    prep::EnzymeForwardOneArgJacobianPrep{B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F,B}
    f_and_df = get_f_and_df(f, backend)
    derivs = jacobian(
        forward_noprimal(backend), f_and_df, x; chunk=Val(B), shadows=prep.shadows
    )
    jac_tensor = only(derivs)
    return maybe_reshape(jac_tensor, prep.output_length, length(x))
end

function DI.value_and_jacobian(
    f::F,
    prep::EnzymeForwardOneArgJacobianPrep{B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F,B}
    f_and_df = get_f_and_df(f, backend)
    (; derivs, val) = jacobian(
        forward_withprimal(backend), f_and_df, x; chunk=Val(B), shadows=prep.shadows
    )
    jac_tensor = only(derivs)
    return val, maybe_reshape(jac_tensor, prep.output_length, length(x))
end

function DI.jacobian!(
    f::F,
    jac,
    prep::EnzymeForwardOneArgJacobianPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F}
    return copyto!(jac, DI.jacobian(f, prep, backend, x))
end

function DI.value_and_jacobian!(
    f::F,
    jac,
    prep::EnzymeForwardOneArgJacobianPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F}
    y, new_jac = DI.value_and_jacobian(f, prep, backend, x)
    return y, copyto!(jac, new_jac)
end
