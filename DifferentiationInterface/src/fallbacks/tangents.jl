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

    ## Tangents{B>1}

    ### 1-arg

    @eval function $prep_op(f::F, backend::AbstractADType, x, seed::Tangents{B}) where {F,B}
        @assert B > 1
        return $prep_op(f, backend, x, Tangents(first(seed)))
    end
    @eval function $op(
        f::F, backend::AbstractADType, x, seed::Tangents{B}, ex::$E
    ) where {F,B}
        @assert B > 1
        resultd = $op.(Ref(f), Ref(backend), Ref(x), Tangents.(seed.d), Ref(ex))
        return Tangents(resultd...)
    end
    @eval function $op!(
        f::F, result, backend::AbstractADType, x, seed::Tangents{B}, ex::$E
    ) where {F,B}
        @assert B > 1
        for b in eacihndex(seed.d, result.d)
            $op!(f, Tangents(result.d[b]), backend, x, Tangents(seed.d[b]), ex)
        end
        return result
    end
    op == :hvp && continue
    @eval function $val_and_op(
        f::F, backend::AbstractADType, x, seed::Tangents{B}, ex::$E
    ) where {F,B}
        @assert B > 1
        result = $op(f, backend, x, seed, ex)
        y = f(x)
        return y, result
    end
    @eval function $val_and_op!(
        f::F, result, backend::AbstractADType, x, seed::Tangents{B}, ex::$E
    ) where {F,B}
        @assert B > 1
        $op!(f, result, backend, x, seed, ex)
        y = f(x)
        return y, result
    end

    ### 2-arg

    @eval function $prep_op(
        f!::F, y, backend::AbstractADType, x, seed::Tangents{B}
    ) where {F,B}
        @assert B > 1
        return $prep_op(f!, y, backend, x, Tangents(first(seed)))
    end
    @eval function $op(
        f!::F, y, backend::AbstractADType, x, seed::Tangents{B}, ex::$E
    ) where {F,B}
        @assert B > 1
        resultd = $op.(Ref(f!), Ref(y), Ref(backend), Ref(x), Tangents.(seed.d), Ref(ex))
        return Tangents(resultd)
    end
    @eval function $op!(
        f!::F, y, result, backend::AbstractADType, x, seed::Tangents{B}, ex::$E
    ) where {F,B}
        @assert B > 1
        for b in eachindex(seed.d, result.d)
            $op!(f!, y, Tangents(result.d[b]), backend, x, Tangents(seed.d[b]), ex)
        end
        return result
    end
    @eval function $val_and_op(
        f!::F, y, backend::AbstractADType, x, seed::Tangents{B}, ex::$E
    ) where {F,B}
        @assert B > 1
        result = $op(f!, y, backend, x, seed, ex)
        f!(y, x)
        return y, result
    end
    @eval function $val_and_op!(
        f!::F, y, result, backend::AbstractADType, x, seed::Tangents{B}, ex::$E
    ) where {F,B}
        @assert B > 1
        $op!(f!, y, result, backend, x, seed, ex)
        f!(y, x)
        return y, result
    end
end
