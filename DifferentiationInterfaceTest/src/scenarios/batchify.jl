function batchify(scen::Scenario{op,args,pl}) where {op,args,pl}
    @compat (; f, x, y, seed, res1, res2) = scen
    if op == :pushforward || op == :pullback
        new_seed = Batch((seed, -seed))
        new_res1 = Batch((res1, -res1))
        return Scenario{op,args,pl}(f; x, y, seed=new_seed, res1=new_res1, res2)
    elseif op == :hvp
        new_seed = Batch((seed, -seed))
        new_res2 = Batch((res2, -res2))
        return Scenario{op,args,pl}(f; x, y, seed=new_seed, res1, res2=new_res2)
    else
        throw(ArgumentError("Scenario $scen is not batchifiable."))
    end
end

function add_batchified!(scens::AbstractVector{<:Scenario})
    batchifiable_scens = filter(s -> operator(s) in (:pushforward, :pullback, :hvp), scens)
    return append!(scens, batchify.(batchifiable_scens))
end
