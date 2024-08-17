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

    ## No Tangents

    ### 1-arg

    @eval function $prep_op(f::F, backend::AbstractADType, x, seed) where {F}
        return $prep_op(f, backend, x, Tangents(seed))
    end
    @eval function $op(f::F, backend::AbstractADType, x, seed, ex::$E) where {F}
        t = $op(f, backend, x, Tangents(seed), ex)
        return only(t)
    end
    @eval function $op!(f::F, result, backend::AbstractADType, x, seed, ex::$E) where {F}
        t = $op!(f, Tangents(result), backend, x, Tangents(seed), ex)
        return only(t)
    end
    op == :hvp && continue
    @eval function $val_and_op(f::F, backend::AbstractADType, x, seed, ex::$E) where {F}
        y, t = $val_and_op(f, backend, x, Tangents(seed), ex)
        return y, only(t)
    end
    @eval function $val_and_op!(
        f::F, result, backend::AbstractADType, x, seed, ex::$E
    ) where {F}
        y, t = $val_and_op!(f, Tangents(result), backend, x, Tangents(seed), ex)
        return y, only(t)
    end

    ### 2-arg

    @eval function $prep_op(f!::F, y, backend::AbstractADType, x, seed) where {F}
        return $prep_op(f!, y, backend, x, Tangents(seed))
    end
    @eval function $op(f!::F, y, backend::AbstractADType, x, seed, ex::$E) where {F}
        t = $op(f!, y, backend, x, Tangents(seed), ex)
        return only(t)
    end
    @eval function $op!(
        f!::F, y, result, backend::AbstractADType, x, seed, ex::$E
    ) where {F}
        t = $op!(f!, y, Tangents(result), backend, x, Tangents(seed), ex)
        return only(t)
    end
    @eval function $val_and_op(f!::F, y, backend::AbstractADType, x, seed, ex::$E) where {F}
        y, t = $val_and_op(f!, y, backend, x, Tangents(seed), ex)
        return y, only(t)
    end
    @eval function $val_and_op!(
        f!::F, y, result, backend::AbstractADType, x, seed, ex::$E
    ) where {F}
        y, t = $val_and_op!(f!, y, Tangents(result), backend, x, Tangents(seed), ex)
        return y, only(t)
    end
end
