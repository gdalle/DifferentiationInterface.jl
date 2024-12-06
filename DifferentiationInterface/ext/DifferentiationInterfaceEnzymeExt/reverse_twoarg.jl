## Pullback

function DI.prepare_pullback(
    f!::F,
    y,
    ::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.NoPullbackPrep()
end

function DI.value_and_pullback(
    f!::F,
    y,
    ::DI.NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    f!_and_df! = get_f_and_df(f!, backend)
    dy_sametype = convert(typeof(y), copy(only(ty)))
    y_and_dy = Duplicated(y, dy_sametype)
    annotated_contexts = translate(backend, Val(1), contexts...)
    dinputs = only(
        autodiff(
            reverse_noprimal(backend),
            f!_and_df!,
            Const,
            y_and_dy,
            Active(x),
            annotated_contexts...,
        ),
    )
    dx = dinputs[2]
    return y, (dx,)
end

function DI.value_and_pullback(
    f!::F,
    y,
    ::DI.NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    f!_and_df! = get_f_and_df(f!, backend, Val(B))
    ty_sametype = map(Fix1(convert, typeof(y)), copy.(ty))
    y_and_ty = BatchDuplicated(y, ty_sametype)
    annotated_contexts = translate(backend, Val(B), contexts...)
    dinputs = only(
        autodiff(
            reverse_noprimal(backend),
            f!_and_df!,
            Const,
            y_and_ty,
            Active(x),
            annotated_contexts...,
        ),
    )
    tx = values(dinputs[2])
    return y, tx
end

function DI.value_and_pullback(
    f!::F,
    y,
    ::DI.NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    f!_and_df! = get_f_and_df(f!, backend)
    dx_sametype = make_zero(x)
    dy_sametype = convert(typeof(y), copy(only(ty)))
    x_and_dx = Duplicated(x, dx_sametype)
    y_and_dy = Duplicated(y, dy_sametype)
    annotated_contexts = translate(backend, Val(1), contexts...)
    autodiff(
        reverse_noprimal(backend),
        f!_and_df!,
        Const,
        y_and_dy,
        x_and_dx,
        annotated_contexts...,
    )
    return y, (dx_sametype,)
end

function DI.value_and_pullback(
    f!::F,
    y,
    ::DI.NoPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    f!_and_df! = get_f_and_df(f!, backend, Val(B))
    tx_sametype = ntuple(_ -> make_zero(x), Val(B))
    ty_sametype = map(Fix1(convert, typeof(y)), copy.(ty))
    x_and_tx = BatchDuplicated(x, tx_sametype)
    y_and_ty = BatchDuplicated(y, ty_sametype)
    annotated_contexts = translate(backend, Val(B), contexts...)
    autodiff(
        reverse_noprimal(backend),
        f!_and_df!,
        Const,
        y_and_ty,
        x_and_tx,
        annotated_contexts...,
    )
    return y, tx_sametype
end
