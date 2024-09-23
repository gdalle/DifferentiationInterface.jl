## Pullback

function DI.prepare_pullback(
    f!::F,
    y,
    ::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents,
    contexts::Vararg{Context,C},
) where {F,C}
    return NoPullbackExtras()
end

function DI.value_and_pullback(
    f!::F,
    y,
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::Tangents{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f!_and_df! = get_f_and_df(f!, backend)
    dy_sametype = convert(typeof(y), copy(only(ty)))
    y_and_dy = Duplicated(y, dy_sametype)
    dinputs = only(
        autodiff(
            reverse_mode_noprimal(backend),
            f!_and_df!,
            Const,
            y_and_dy,
            Active(x),
            map(translate, contexts)...,
        ),
    )
    dx = dinputs[2]
    return y, Tangents(dx)
end

function DI.value_and_pullback(
    f!::F,
    y,
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f!_and_df! = get_f_and_df(f!, backend, Val(B))
    dys_sametype = map(Fix1(convert, typeof(y)), copy.(ty.d))
    y_and_dys = BatchDuplicated(y, dys_sametype)
    dinputs = only(
        autodiff(
            reverse_mode_noprimal(backend),
            f!_and_df!,
            Const,
            y_and_dys,
            Active(x),
            map(translate, contexts)...,
        ),
    )
    dxs = dinputs[2]
    return y, Tangents(dxs...)
end

function DI.value_and_pullback(
    f!::F,
    y,
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{1},
    contexts::Vararg{Context,C},
) where {F,C}
    f!_and_df! = get_f_and_df(f!, backend)
    dx_sametype = make_zero(x)
    dy_sametype = convert(typeof(y), copy(only(ty)))
    x_and_dx = Duplicated(x, dx_sametype)
    y_and_dy = Duplicated(y, dy_sametype)
    autodiff(
        reverse_mode_noprimal(backend),
        f!_and_df!,
        Const,
        y_and_dy,
        x_and_dx,
        map(translate, contexts)...,
    )
    return y, Tangents(dx_sametype)
end

function DI.value_and_pullback(
    f!::F,
    y,
    ::NoPullbackExtras,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::Tangents{B},
    contexts::Vararg{Context,C},
) where {F,B,C}
    f!_and_df! = get_f_and_df(f!, backend, Val(B))
    dxs_sametype = ntuple(_ -> make_zero(x), Val(B))
    dys_sametype = map(Fix1(convert, typeof(y)), copy.(ty.d))
    x_and_dxs = BatchDuplicated(x, dxs_sametype)
    y_and_dys = BatchDuplicated(y, dys_sametype)
    autodiff(
        reverse_mode_noprimal(backend),
        f!_and_df!,
        Const,
        y_and_dys,
        x_and_dxs,
        map(translate, contexts)...,
    )
    return y, Tangents(dxs_sametype...)
end
