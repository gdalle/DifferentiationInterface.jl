## Traits

for trait in (
    :check_available,
    :mutation_support,
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
    E = if op == :pushforward
        :PushforwardExtras
    elseif op == :pullback
        :PullbackExtras
    elseif op == :hvp
        :HVPExtras
    end

    ## One argument
    @eval begin
        $prep(f, ba::AutoSparse, x, v) = $prep(f, dense_ad(ba), x, v)
        $op(f, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) =
            $op(f, dense_ad(ba), x, v, ex)
        $valop(f, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) =
            $valop(f, dense_ad(ba), x, v, ex)
        $op!(f, res, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) =
            $op!(f, res, dense_ad(ba), x, v, ex)
        $valop!(f, res, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) =
            $valop!(f, res, dense_ad(ba), x, v, ex)
    end

    ## Two arguments
    @eval begin
        $prep(f!, y, ba::AutoSparse, x, v) = $prep(f!, y, dense_ad(ba), x, v)
        $op(f!, y, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) =
            $op(f!, y, dense_ad(ba), x, v, ex)
        $valop(f!, y, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) =
            $valop(f!, y, dense_ad(ba), x, v, ex)
        $op!(f!, y, res, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) =
            $op!(f!, y, res, dense_ad(ba), x, v, ex)
        $valop!(f!, y, res, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) =
            $valop!(f!, y, res, dense_ad(ba), x, v, ex)
    end

    ## Split
    if op == :pullback
        valop_split = Symbol("value_and_", op, "_split")
        valop!_split = Symbol("value_and_", op!, "_split")

        @eval begin
            $valop_split(f, ba::AutoSparse, x, ex::$E=$prep(f, ba, x, f(x))) =
                $valop_split(f, dense_ad(ba), x, ex)
            $valop!_split(f, ba::AutoSparse, x, ex::$E=$prep(f, ba, x, f(x))) =
                $valop!_split(f, dense_ad(ba), x, ex)
            $valop_split(f!, y, ba::AutoSparse, x, ex::$E=$prep(f, ba, x, similar(y))) =
                $valop_split(f!, y, dense_ad(ba), x, ex)
            $valop!_split(f!, y, ba::AutoSparse, x, ex::$E=$prep(f, ba, x, similar(y))) =
                $valop!_split(f!, y, dense_ad(ba), x, ex)
        end
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
        $prep(f, ba::AutoSparse, x) = $prep(f, dense_ad(ba), x)
        $op(f, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) = $op(f, dense_ad(ba), x, ex)
        $valop(f, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) =
            $valop(f, dense_ad(ba), x, ex)
        $op!(f, res, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) =
            $op!(f, res, dense_ad(ba), x, ex)
        $valop!(f, res, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) =
            $valop!(f, res, dense_ad(ba), x, ex)
    end

    ## Two arguments
    if op in (:derivative, :jacobian)
        @eval begin
            $prep(f!, y, ba::AutoSparse, x) = $prep(f!, y, dense_ad(ba), x)
            $op(f!, y, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) =
                $op(f!, y, dense_ad(ba), x, ex)
            $valop(f!, y, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) =
                $valop(f!, y, dense_ad(ba), x, ex)
            $op!(f!, y, res, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) =
                $op!(f!, y, res, dense_ad(ba), x, ex)
            $valop!(f!, y, res, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) =
                $valop!(f!, y, res, dense_ad(ba), x, ex)
        end
    end
end
