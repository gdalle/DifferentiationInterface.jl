function DI.prepare_hvp(f, ::AnyAutoEnzyme{Nothing,Nothing}, x, tx::Tangents{1})
    return NoHVPExtras()
end

function DI.hvp(f, ::AnyAutoEnzyme{Nothing,Nothing}, x, tx::Tangents{1}, ::NoHVPExtras)
    return SingleTangent(hvp(f, x, only(tx)))
end

function DI.hvp!(
    f, tg::Tangents{1}, ::AnyAutoEnzyme{Nothing,Nothing}, x, tx::Tangents{1}, ::NoHVPExtras
)
    hvp!(only(tg), f, x, only(tx))
    return tg
end
