for op in (:pushforward, :pullback, :hvp)
    op! = Symbol(op, "!")
    val_prefix = "value_and_"
    val_and_op = Symbol(val_prefix, op)
    val_and_op! = Symbol(val_prefix, op!)
    prep_op = Symbol("prepare_", op)
    prep_op_same_point = Symbol("prepare_", op, "_same_point")

    E = if op == :pushforward
        PushforwardExtras
    elseif op == :pullback
        PullbackExtras
    elseif op == :hvp
        HVPExtras
    end

    ### 1-arg

    @eval function $prep_op(f::F, backend::AnyAutoEnzyme, x, seed::Tangents) where {F}
        return $prep_op(f, backend, x, Tangents(first(seed)))
    end
    @eval function $op(f::F, backend::AnyAutoEnzyme, x, seed::Tangents, ex::$E) where {F}
        resultd = $op.(Ref(f), Ref(backend), Ref(x), Tangents.(seed.d), Ref(ex))
        return Tangents(resultd...)
    end
    @eval function $op!(
        f::F, result, backend::AnyAutoEnzyme, x, seed::Tangents, ex::$E
    ) where {F}
        for b in eachindex(seed.d, result.d)
            $op!(f, Tangents(result.d[b]), backend, x, Tangents(seed.d[b]), ex)
        end
        return result
    end
    op == :hvp && continue
    @eval function $val_and_op(
        f::F, backend::AnyAutoEnzyme, x, seed::Tangents, ex::$E
    ) where {F}
        result = $op(f, backend, x, seed, ex)
        y = f(x)
        return y, result
    end
    @eval function $val_and_op!(
        f::F, result, backend::AnyAutoEnzyme, x, seed::Tangents, ex::$E
    ) where {F}
        $op!(f, result, backend, x, seed, ex)
        y = f(x)
        return y, result
    end

    ### 2-arg

    @eval function $prep_op(f!::F, y, backend::AnyAutoEnzyme, x, seed::Tangents) where {F}
        return $prep_op(f!, y, backend, x, Tangents(first(seed)))
    end
    @eval function $op(
        f!::F, y, backend::AnyAutoEnzyme, x, seed::Tangents, ex::$E
    ) where {F}
        resultd = $op.(Ref(f!), Ref(y), Ref(backend), Ref(x), Tangents.(seed.d), Ref(ex))
        return Tangents(resultd)
    end
    @eval function $op!(
        f!::F, y, result, backend::AnyAutoEnzyme, x, seed::Tangents, ex::$E
    ) where {F}
        for b in eachindex(seed.d, result.d)
            $op!(f!, y, Tangents(result.d[b]), backend, x, Tangents(seed.d[b]), ex)
        end
        return result
    end
    @eval function $val_and_op(
        f!::F, y, backend::AnyAutoEnzyme, x, seed::Tangents, ex::$E
    ) where {F}
        result = $op(f!, y, backend, x, seed, ex)
        f!(y, x)
        return y, result
    end
    @eval function $val_and_op!(
        f!::F, y, result, backend::AnyAutoEnzyme, x, seed::Tangents, ex::$E
    ) where {F}
        $op!(f!, y, result, backend, x, seed, ex)
        f!(y, x)
        return y, result
    end
end
