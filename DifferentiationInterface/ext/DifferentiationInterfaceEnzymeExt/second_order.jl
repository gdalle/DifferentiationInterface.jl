function DI.prepare_hvp(
    f, ::AnyAutoEnzyme{Nothing,<:Union{Nothing,Const}}, x, tx::Tangents{1}
)
    return NoHVPExtras()
end

function DI.hvp(
    f, ::AnyAutoEnzyme{Nothing,<:Union{Nothing,Const}}, x, tx::Tangents{1}, ::NoHVPExtras
)
    f_and_df = get_f_and_df(f, backend)
    return SingleTangent(hvp(f_and_df, x, only(tx)))
end

function DI.hvp!(
    f,
    tg::Tangents{1},
    ::AnyAutoEnzyme{Nothing,<:Union{Nothing,Const}},
    x,
    tx::Tangents{1},
    ::NoHVPExtras,
)
    f_and_df = get_f_and_df(f, backend)
    hvp!(only(tg), f_and_df, x, only(tx))
    return tg
end
