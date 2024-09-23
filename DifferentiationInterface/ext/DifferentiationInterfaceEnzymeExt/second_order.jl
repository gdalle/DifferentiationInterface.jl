function DI.prepare_hvp(f, ::AutoEnzyme{Nothing,Nothing}, x, tx::Tangents{1})
    return NoHVPExtras()
end

function DI.hvp(f, ::NoHVPExtras, ::AutoEnzyme{Nothing,Nothing}, x, tx::Tangents{1})
    return Tangents(hvp(f, x, only(tx)))
end

function DI.hvp!(
    f, tg::Tangents{1}, ::NoHVPExtras, ::AutoEnzyme{Nothing,Nothing}, x, tx::Tangents{1}
)
    hvp!(only(tg), f, x, only(tx))
    return tg
end
