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
        @assert !isa(seed, Tangents)
        return $prep_op(f, backend, x, SingleTangent(seed))
    end
    @eval function $op(f::F, ex::$E, backend::AbstractADType, x, seed) where {F}
        @assert !isa(seed, Tangents)
        t = $op(f, ex, backend, x, SingleTangent(seed))
        return only(t)
    end
    @eval function $op!(f::F, result, ex::$E, backend::AbstractADType, x, seed) where {F}
        @assert !isa(seed, Tangents) && !isa(result, Tangents)
        t = $op!(f, SingleTangent(result), ex, backend, x, SingleTangent(seed))
        return only(t)
    end
    op == :hvp && continue
    @eval function $val_and_op(f::F, ex::$E, backend::AbstractADType, x, seed) where {F}
        @assert !isa(seed, Tangents)
        y, t = $val_and_op(f, ex, backend, x, SingleTangent(seed))
        return y, only(t)
    end
    @eval function $val_and_op!(
        f::F, result, ex::$E, backend::AbstractADType, x, seed
    ) where {F}
        @assert !isa(seed, Tangents) && !isa(result, Tangents)
        y, t = $val_and_op!(f, SingleTangent(result), ex, backend, x, SingleTangent(seed))
        return y, only(t)
    end

    ### 2-arg

    @eval function $prep_op(f!::F, y, backend::AbstractADType, x, seed) where {F}
        @assert !isa(seed, Tangents)
        return $prep_op(f!, y, backend, x, SingleTangent(seed))
    end
    @eval function $op(f!::F, y, ex::$E, backend::AbstractADType, x, seed) where {F}
        @assert !isa(seed, Tangents)
        t = $op(f!, y, ex, backend, x, SingleTangent(seed))
        return only(t)
    end
    @eval function $op!(
        f!::F, y, result, ex::$E, backend::AbstractADType, x, seed
    ) where {F}
        @assert !isa(seed, Tangents) && !isa(result, Tangents)
        t = $op!(f!, y, SingleTangent(result), ex, backend, x, SingleTangent(seed))
        return only(t)
    end
    @eval function $val_and_op(f!::F, y, ex::$E, backend::AbstractADType, x, seed) where {F}
        @assert !isa(seed, Tangents)
        y, t = $val_and_op(f!, y, ex, backend, x, SingleTangent(seed))
        return y, only(t)
    end
    @eval function $val_and_op!(
        f!::F, y, result, ex::$E, backend::AbstractADType, x, seed
    ) where {F}
        @assert !isa(seed, Tangents) && !isa(result, Tangents)
        y, t = $val_and_op!(
            f!, y, SingleTangent(result), ex, backend, x, SingleTangent(seed)
        )
        return y, only(t)
    end
end
