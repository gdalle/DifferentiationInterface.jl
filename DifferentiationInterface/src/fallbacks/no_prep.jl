for op in (:derivative, :gradient, :jacobian)
    op! = Symbol(op, "!")
    val_prefix = "value_and_"
    val_and_op = Symbol(val_prefix, op)
    val_and_op! = Symbol(val_prefix, op!)
    prep_op = Symbol("prepare_", op)
    # 1-arg
    @eval function $op(
        f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, contexts...)
        return $op(f, prep, backend, x, contexts...)
    end
    @eval function $op!(
        f::F, result, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, contexts...)
        return $op!(f, result, prep, backend, x, contexts...)
    end
    @eval function $val_and_op(
        f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, contexts...)
        return $val_and_op(f, prep, backend, x, contexts...)
    end
    @eval function $val_and_op!(
        f::F, result, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, contexts...)
        return $val_and_op!(f, result, prep, backend, x, contexts...)
    end
    op == :gradient && continue
    # 2-arg
    @eval function $op(
        f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f!, y, backend, x, contexts...)
        return $op(f!, y, prep, backend, x, contexts...)
    end
    @eval function $op!(
        f!::F, y, result, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f!, y, backend, x, contexts...)
        return $op!(f!, y, result, prep, backend, x, contexts...)
    end
    @eval function $val_and_op(
        f!::F, y, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f!, y, backend, x, contexts...)
        return $val_and_op(f!, y, prep, backend, x, contexts...)
    end
    @eval function $val_and_op!(
        f!::F, y, result, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f!, y, backend, x, contexts...)
        return $val_and_op!(f!, y, result, prep, backend, x, contexts...)
    end
end

for op in (:second_derivative, :hessian)
    op! = Symbol(op, "!")
    val_prefix = if op == :second_derivative
        "value_derivative_and_"
    elseif op == :hessian
        "value_gradient_and_"
    end
    val_and_op = Symbol(val_prefix, op)
    val_and_op! = Symbol(val_prefix, op!)
    prep_op = Symbol("prepare_", op)
    # 1-arg
    @eval function $op(
        f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, contexts...)
        return $op(f, prep, backend, x, contexts...)
    end
    @eval function $op!(
        f::F, result2, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, contexts...)
        return $op!(f, result2, prep, backend, x, contexts...)
    end
    @eval function $val_and_op(
        f::F, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, contexts...)
        return $val_and_op(f, prep, backend, x, contexts...)
    end
    @eval function $val_and_op!(
        f::F, result1, result2, backend::AbstractADType, x, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, contexts...)
        return $val_and_op!(f, result1, result2, prep, backend, x, contexts...)
    end
end

for op in (:pushforward, :pullback, :hvp)
    op! = Symbol(op, "!")
    val_prefix = "value_and_"
    val_and_op = Symbol(val_prefix, op)
    val_and_op! = Symbol(val_prefix, op!)
    prep_op = Symbol("prepare_", op)
    prep_op_same_point = Symbol("prepare_", op, "_same_point")
    E = if startswith(string(op), "pushforward")
        PushforwardPrep
    elseif startswith(string(op), "pullback")
        PullbackPrep
    elseif startswith(string(op), "hvp")
        HVPPrep
    end
    # 1-arg
    @eval function $prep_op_same_point(
        f::F, backend::AbstractADType, x, seed::Tangents, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, seed, contexts...)
        return $prep_op_same_point(f, prep, backend, x, seed, contexts...)
    end
    @eval function $prep_op_same_point(
        f::F,
        prep::$E,
        backend::AbstractADType,
        x,
        seed::Tangents,
        contexts::Vararg{Context,C},
    ) where {F,C}
        return prep
    end
    @eval function $op(
        f::F, backend::AbstractADType, x, seed::Tangents, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, seed, contexts...)
        return $op(f, prep, backend, x, seed, contexts...)
    end
    @eval function $op!(
        f::F,
        result::Tangents,
        backend::AbstractADType,
        x,
        seed::Tangents,
        contexts::Vararg{Context,C},
    ) where {F,C}
        prep = $prep_op(f, backend, x, seed, contexts...)
        return $op!(f, result, prep, backend, x, seed, contexts...)
    end
    op == :hvp && continue
    @eval function $val_and_op(
        f::F, backend::AbstractADType, x, seed::Tangents, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f, backend, x, seed, contexts...)
        return $val_and_op(f, prep, backend, x, seed, contexts...)
    end
    @eval function $val_and_op!(
        f::F,
        result::Tangents,
        backend::AbstractADType,
        x,
        seed::Tangents,
        contexts::Vararg{Context,C},
    ) where {F,C}
        prep = $prep_op(f, backend, x, seed, contexts...)
        return $val_and_op!(f, result, prep, backend, x, seed, contexts...)
    end
    # 2-arg
    @eval function $prep_op_same_point(
        f!::F, y, backend::AbstractADType, x, seed::Tangents, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f!, y, backend, x, seed, contexts...)
        return $prep_op_same_point(f!, y, prep, backend, x, seed, contexts...)
    end
    @eval function $prep_op_same_point(
        f!::F,
        y,
        prep::$E,
        backend::AbstractADType,
        x,
        seed::Tangents,
        contexts::Vararg{Context,C},
    ) where {F,C}
        return prep
    end
    @eval function $op(
        f!::F, y, backend::AbstractADType, x, seed::Tangents, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f!, y, backend, x, seed, contexts...)
        return $op(f!, y, prep, backend, x, seed, contexts...)
    end
    @eval function $op!(
        f!::F,
        y,
        result::Tangents,
        backend::AbstractADType,
        x,
        seed::Tangents,
        contexts::Vararg{Context,C},
    ) where {F,C}
        prep = $prep_op(f!, y, backend, x, seed, contexts...)
        return $op!(f!, y, result, prep, backend, x, seed, contexts...)
    end
    @eval function $val_and_op(
        f!::F, y, backend::AbstractADType, x, seed::Tangents, contexts::Vararg{Context,C}
    ) where {F,C}
        prep = $prep_op(f!, y, backend, x, seed, contexts...)
        return $val_and_op(f!, y, prep, backend, x, seed, contexts...)
    end
    @eval function $val_and_op!(
        f!::F,
        y,
        result::Tangents,
        backend::AbstractADType,
        x,
        seed::Tangents,
        contexts::Vararg{Context,C},
    ) where {F,C}
        prep = $prep_op(f!, y, backend, x, seed, contexts...)
        return $val_and_op!(f!, y, result, prep, backend, x, seed, contexts...)
    end
end
