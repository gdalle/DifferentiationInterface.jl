## Pushforward

function DI.prepare_pushforward(
    f, ::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}}, x, tx::Tangents
)
    return NoPushforwardExtras()
end

function DI.value_and_pushforward(
    f,
    extras::NoPushforwardExtras,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
)
    ty = map(tx) do dx
        only(DI.pushforward(f, extras, backend, x, Tangents(dx)))
    end
    y = f(x)
    return y, ty
end

function DI.value_and_pushforward(
    f,
    ::NoPushforwardExtras,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{1},
)
    dx = only(tx)
    f_and_df = get_f_and_df(f, backend)
    dx_sametype = convert(typeof(x), dx)
    x_and_dx = Duplicated(x, dx_sametype)
    new_dy, y = if backend isa AutoDeferredEnzyme
        values(autodiff_deferred(mode_withprimal(backend), f_and_df, x_and_dx))
    else
        values(autodiff(mode_withprimal(backend), f_and_df, x_and_dx))
    end
    return y, Tangents(new_dy)
end

function DI.pushforward(
    f,
    ::NoPushforwardExtras,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents{1},
)
    dx = only(tx)
    f_and_df = get_f_and_df(f, backend)
    dx_sametype = convert(typeof(x), dx)
    x_and_dx = Duplicated(x, dx_sametype)
    new_dy = if backend isa AutoDeferredEnzyme
        only(autodiff_deferred(mode_noprimal(backend), f_and_df, x_and_dx))
    else
        only(autodiff(mode_noprimal(backend), f_and_df, x_and_dx))
    end
    return Tangents(new_dy)
end

function DI.value_and_pushforward!(
    f,
    ty::Tangents,
    extras::NoPushforwardExtras,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
)
    # dy cannot be passed anyway
    y, new_ty = DI.value_and_pushforward(f, extras, backend, x, tx)
    return y, copyto!(ty, new_ty)
end

function DI.pushforward!(
    f,
    ty::Tangents,
    extras::NoPushforwardExtras,
    backend::AnyAutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::Tangents,
)
    # dy cannot be passed anyway
    return copyto!(ty, DI.pushforward(f, extras, backend, x, tx))
end

## Gradient

struct EnzymeForwardGradientExtras{B,O} <: GradientExtras
    shadows::O
end

function DI.prepare_gradient(
    f, backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}}, x
)
    B = pick_batchsize(backend, length(x))
    shadows = create_shadows(Val(B), x)
    return EnzymeForwardGradientExtras{B,typeof(shadows)}(shadows)
end

function DI.gradient(
    f,
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {B}
    f_and_df = get_f_and_df(f, backend)
    gradient(mode_noprimal(backend), f_and_df, x; chunk=Val(B), shadows=extras.shadows)[1]
end

function DI.value_and_gradient(
    f,
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {B}
    f_and_df = get_f_and_df(f, backend)
    gr = gradient(mode_withprimal(backend), f_and_df, x; chunk=Val(B), shadows=extras.shadows)
    return gr.val, gr.derivs[1]
end

function DI.gradient!(
    f,
    grad,
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {B}
    f_and_df = get_f_and_df(f, backend)
    gr = gradient(mode_noprimal(backend), f_and_df, x; chunk=Val(B), shadows=extras.shadows)[1]
    return copyto!(grad, gr)
end

function DI.value_and_gradient!(
    f,
    grad,
    extras::EnzymeForwardGradientExtras{B},
    backend::AutoEnzyme{<:ForwardMode,<:Union{Nothing,Const}},
    x,
) where {B}
    f_and_df = get_f_and_df(f, backend)
    gr = gradient(mode_withprimal(backend), f_and_df, x, Val(B); shadows=extras.shadow)
    return gr.val, copyto!(grad, gr.derivs[1])
end

## Jacobian

struct EnzymeForwardOneArgJacobianExtras{B,O} <: JacobianExtras
    shadows::O
end

function DI.prepare_jacobian(
    f, backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}}, x
)
    B = pick_batchsize(backend, length(x))
    shadows = create_shadows(Val(B), x)
    return EnzymeForwardOneArgJacobianExtras{B,typeof(shadows)}(shadows)
end

function DI.jacobian(
    f,
    extras::EnzymeForwardOneArgJacobianExtras{B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {B}
    f_and_df = get_f_and_df(f, backend)
    jacobian(mode_noprimal(backend), f_and_df, x; chunk=Val(B), shadows=extras.shadows)[1]
end

function DI.value_and_jacobian(
    f,
    extras::EnzymeForwardOneArgJacobianExtras{B},
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
) where {B}
    f_and_df = get_f_and_df(f, backend)
    jac = jacobian(mode_withprimal(backend), f_and_df, x; chunk=Val(B), shadows=extras.shadows)
    return jac.val, jac.derivs[1]
end

function DI.jacobian!(
    f,
    jac,
    extras::EnzymeForwardOneArgJacobianExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    return copyto!(jac, DI.jacobian(f, extras, backend, x))
end

function DI.value_and_jacobian!(
    f,
    jac,
    extras::EnzymeForwardOneArgJacobianExtras,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing},<:Union{Nothing,Const}},
    x,
)
    y, new_jac = DI.value_and_jacobian(f, extras, backend, x)
    return y, copyto!(jac, new_jac)
end
