## Pushforward

function DI.prepare_pushforward(
    f::F,
    ::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return NoPushforwardExtras()
end

function DI.value_and_pushforward(
    f::F,
    ::NoPushforwardExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    dx_sametype = convert(typeof(x), only(tx))
    x_and_dx = Duplicated(x, dx_sametype)
    dy, y = autodiff(
        forward_mode_withprimal(backend), f_and_df, x_and_dx, map(translate, contexts)...
    )
    return y, Tangents(dy)
end

function DI.value_and_pushforward(
    f::F,
    ::NoPushforwardExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f_and_df = get_f_and_df(f, backend, Val(B))
    dxs_sametype = map(Fix1(convert, typeof(x)), tx.d)
    x_and_dxs = BatchDuplicated(x, dxs_sametype)
    dys, y = autodiff(
        forward_mode_withprimal(backend), f_and_df, x_and_dxs, map(translate, contexts)...
    )
    return y, Tangents(dys...)
end

function DI.pushforward(
    f::F,
    ::NoPushforwardExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f_and_df = get_f_and_df(f, backend)
    dx_sametype = convert(typeof(x), only(tx))
    x_and_dx = Duplicated(x, dx_sametype)
    dy = only(
        autodiff(
            forward_mode_noprimal(backend), f_and_df, x_and_dx, map(translate, contexts)...
        ),
    )
    return Tangents(dy)
end

function DI.pushforward(
    f::F,
    ::NoPushforwardExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f_and_df = get_f_and_df(f, backend, Val(B))
    dxs_sametype = map(Fix1(convert, typeof(x)), tx.d)
    x_and_dxs = BatchDuplicated(x, dxs_sametype)
    dys = only(
        autodiff(
            forward_mode_noprimal(backend), f_and_df, x_and_dxs, map(translate, contexts)...
        ),
    )
    return Tangents(dys...)
end

function DI.value_and_pushforward!(
    f::F,
    ty::Tangents,
    extras::NoPushforwardExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    # dy cannot be passed anyway
    y, new_ty = DI.value_and_pushforward(f, extras, backend, x, tx, contexts...)
    return y, copyto!(ty, new_ty)
end

function DI.pushforward!(
    f::F,
    ty::Tangents,
    extras::NoPushforwardExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    # dy cannot be passed anyway
    return copyto!(ty, DI.pushforward(f, extras, backend, x, tx, contexts...))
end

## Gradient

struct EnzymeForwardGradientExtras{B,O} <: GradientExtras
    shadows::O
end

function DI.prepare_gradient(
    f::F, backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}}, x
) where {F}
    B = pick_batchsize(backend, length(x))
    shadows = create_shadows(Val(B), x)
    return EnzymeForwardGradientExtras{B,typeof(shadows)}(shadows)
end

function DI.gradient(
    f::F,
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    f_and_df = get_f_and_df(f, backend)
    derivs = gradient(
        forward_mode_noprimal(backend), f_and_df, x; chunk=Val(B), shadows=extras.shadows
    )
    return only(derivs)
end

function DI.value_and_gradient(
    f::F,
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    f_and_df = get_f_and_df(f, backend)
    (; derivs, val) = gradient(
        forward_mode_withprimal(backend), f_and_df, x; chunk=Val(B), shadows=extras.shadows
    )
    return val, only(derivs)
end

function DI.gradient!(
    f::F,
    grad,
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    return copyto!(grad, DI.gradient(f, extras, backend, x))
end

function DI.value_and_gradient!(
    f::F,
    grad,
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {F,B}
    y, new_grad = DI.value_and_gradient(f, extras, backend, x)
    return y, copyto!(grad, new_grad)
end

## Jacobian

struct EnzymeForwardOneArgJacobianExtras{B,O} <: JacobianExtras
    shadows::O
    output_length::Int
end

function DI.prepare_jacobian(
    f::F, backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}}, x
) where {F}
    y = f(x)
    B = pick_batchsize(backend, length(x))
    shadows = create_shadows(Val(B), x)
    return EnzymeForwardOneArgJacobianExtras{B,typeof(shadows)}(shadows, length(y))
end

function DI.jacobian(
    f::F,
    extras::EnzymeForwardOneArgJacobianExtras{B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F,B}
    f_and_df = get_f_and_df(f, backend)
    derivs = jacobian(
        forward_mode_noprimal(backend), f_and_df, x; chunk=Val(B), shadows=extras.shadows
    )
    jac_tensor = only(derivs)
    return maybe_reshape(jac_tensor, extras.output_length, length(x))
end

function DI.value_and_jacobian(
    f::F,
    extras::EnzymeForwardOneArgJacobianExtras{B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F,B}
    f_and_df = get_f_and_df(f, backend)
    (; derivs, val) = jacobian(
        forward_mode_withprimal(backend), f_and_df, x; chunk=Val(B), shadows=extras.shadows
    )
    jac_tensor = only(derivs)
    return val, maybe_reshape(jac_tensor, extras.output_length, length(x))
end

function DI.jacobian!(
    f::F,
    jac,
    extras::EnzymeForwardOneArgJacobianExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F}
    return copyto!(jac, DI.jacobian(f, extras, backend, x))
end

function DI.value_and_jacobian!(
    f::F,
    jac,
    extras::EnzymeForwardOneArgJacobianExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {F}
    y, new_jac = DI.value_and_jacobian(f, extras, backend, x)
    return y, copyto!(jac, new_jac)
end
