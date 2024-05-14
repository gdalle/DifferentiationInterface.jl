## Traits

for trait in (
    :check_available,
    :twoarg_support,
    :pushforward_performance,
    :pullback_performance,
    :hvp_mode,
)
    @eval $trait(backend::AutoSparse) = $trait(dense_ad(backend))
end

## Operators

for op in (:pushforward, :pullback, :hvp)
    op! = Symbol(op, "!")
    valop = Symbol("value_and_", op)
    valop! = Symbol("value_and_", op, "!")
    prep = Symbol("prepare_", op)
    prepsame = Symbol("prepare_", op, "_same_point")
    E = if op == :pushforward
        :PushforwardExtras
    elseif op == :pullback
        :PullbackExtras
    elseif op == :hvp
        :HVPExtras
    end

    ## One argument
    @eval begin
        $prep(f::F, ba::AutoSparse, x, v) where {F} = $prep(f, dense_ad(ba), x, v)
        $prepsame(f::F, ba::AutoSparse, x, v) where {F} = $prepsame(f, dense_ad(ba), x, v)
        $prepsame(f::F, ba::AutoSparse, x, v, ex::$E) where {F} =
            $prepsame(f, dense_ad(ba), x, v, ex)
        $op(f::F, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) where {F} =
            $op(f, dense_ad(ba), x, v, ex)
        $valop(f::F, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) where {F} =
            $valop(f, dense_ad(ba), x, v, ex)
        $op!(f::F, res, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) where {F} =
            $op!(f, res, dense_ad(ba), x, v, ex)
        $valop!(f::F, res, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) where {F} =
            $valop!(f, res, dense_ad(ba), x, v, ex)
    end

    ## Two arguments
    @eval begin
        $prep(f!::F, y, ba::AutoSparse, x, v) where {F} = $prep(f!, y, dense_ad(ba), x, v)
        $prepsame(f!::F, y, ba::AutoSparse, x, v) where {F} =
            $prepsame(f!, y, dense_ad(ba), x, v)
        $prepsame(f!::F, y, ba::AutoSparse, x, v, ex::$E) where {F} =
            $prepsame(f!, y, dense_ad(ba), x, v, ex)
        $op(f!::F, y, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) where {F} =
            $op(f!, y, dense_ad(ba), x, v, ex)
        $valop(f!::F, y, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) where {F} =
            $valop(f!, y, dense_ad(ba), x, v, ex)
        $op!(f!::F, y, res, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) where {F} =
            $op!(f!, y, res, dense_ad(ba), x, v, ex)
        $valop!(
            f!::F, y, res, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)
        ) where {F} = $valop!(f!, y, res, dense_ad(ba), x, v, ex)
    end
end

for op in (:derivative, :gradient, :second_derivative)
    op! = Symbol(op, "!")
    valop = Symbol("value_and_", op)
    valop! = Symbol("value_and_", op, "!")
    prep = Symbol("prepare_", op)
    E = if op == :derivative
        :DerivativeExtras
    elseif op == :gradient
        :GradientExtras
    elseif op == :second_derivative
        :SecondDerivativeExtras
    end

    ## One argument
    @eval begin
        $prep(f::F, ba::AutoSparse, x) where {F} = $prep(f, dense_ad(ba), x)
        $op(f::F, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) where {F} =
            $op(f, dense_ad(ba), x, ex)
        $valop(f::F, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) where {F} =
            $valop(f, dense_ad(ba), x, ex)
        $op!(f::F, res, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) where {F} =
            $op!(f, res, dense_ad(ba), x, ex)
        $valop!(f::F, res, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) where {F} =
            $valop!(f, res, dense_ad(ba), x, ex)
    end

    ## Two arguments
    if op in (:derivative,)
        @eval begin
            $prep(f!::F, y, ba::AutoSparse, x) where {F} = $prep(f!, y, dense_ad(ba), x)
            $op(f!::F, y, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) where {F} =
                $op(f!, y, dense_ad(ba), x, ex)
            $valop(f!::F, y, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) where {F} =
                $valop(f!, y, dense_ad(ba), x, ex)
            $op!(f!::F, y, res, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) where {F} =
                $op!(f!, y, res, dense_ad(ba), x, ex)
            $valop!(
                f!::F, y, res, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)
            ) where {F} = $valop!(f!, y, res, dense_ad(ba), x, ex)
        end
    end
end
