for op in [
    :derivative,
    :gradient,
    :hessian,
    :hvp,
    :jacobian,
    :pullback,
    :pushforward,
    :second_derivative,
]
    op! = Symbol(op, "!")
    val_prefix = if op == :second_derivative
        "value_derivative_and_"
    elseif op in [:hessian, :hvp]
        "value_gradient_and_"
    else
        "value_and_"
    end
    val_and_op = Symbol(val_prefix, op)
    val_and_op! = Symbol(val_prefix, op!)
    prep_op = Symbol("prepare_", op)

    S1out = Scenario{op,:out,:out}
    S1in = Scenario{op,:in,:out}
    S2out = Scenario{op,:out,:in}
    S2in = Scenario{op,:in,:in}

    if op in [:derivative, :gradient, :jacobian]
        @eval function test_jet(ba::AbstractADType, scen::$S1out)
            @compat (; f, x) = deepcopy(scen)
            ex = $prep_op(f, ba, x)
            JET.@test_opt $op(f, ex, ba, x)
            JET.@test_call $op(f, ex, ba, x)
            JET.@test_opt $val_and_op(f, ex, ba, x)
            JET.@test_call $val_and_op(f, ex, ba, x)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, res1) = deepcopy(scen)
            ex = $prep_op(f, ba, x)
            JET.@test_opt $op!(f, res1, ex, ba, x)
            JET.@test_call $op!(f, res1, ex, ba, x)
            JET.@test_opt $val_and_op!(f, res1, ex, ba, x)
            JET.@test_call $val_and_op!(f, res1, ex, ba, x)
        end

        op == :gradient && continue

        @eval function test_jet(ba::AbstractADType, scen::$S2out)
            @compat (; f, x, y) = deepcopy(scen)
            ex = $prep_op(f, y, ba, x)
            JET.@test_opt $op(f, y, ex, ba, x)
            JET.@test_call $op(f, y, ex, ba, x)
            JET.@test_opt $val_and_op(f, y, ex, ba, x)
            JET.@test_call $val_and_op(f, y, ex, ba, x)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S2in)
            @compat (; f, x, y, res1) = deepcopy(scen)
            ex = $prep_op(f, y, ba, x)
            JET.@test_opt $op!(f, y, res1, ex, ba, x)
            JET.@test_call $op!(f, y, res1, ex, ba, x)
            JET.@test_opt $val_and_op!(f, y, res1, ex, ba, x)
            JET.@test_call $val_and_op!(f, y, res1, ex, ba, x)
        end

    elseif op in [:second_derivative, :hessian]
        @eval function test_jet(ba::AbstractADType, scen::$S1out)
            @compat (; f, x) = deepcopy(scen)
            ex = $prep_op(f, ba, x)
            JET.@test_opt $op(f, ex, ba, x)
            JET.@test_call $op(f, ex, ba, x)
            JET.@test_opt $val_and_op(f, ex, ba, x)
            JET.@test_call $val_and_op(f, ex, ba, x)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, res1, res2) = deepcopy(scen)
            ex = $prep_op(f, ba, x)
            JET.@test_opt $op!(f, res2, ex, ba, x)
            JET.@test_call $op!(f, res2, ex, ba, x)
            JET.@test_opt $val_and_op!(f, res1, res2, ex, ba, x)
            JET.@test_call $val_and_op!(f, res1, res2, ex, ba, x)
        end

    elseif op in [:pushforward, :pullback]
        @eval function test_jet(ba::AbstractADType, scen::$S1out)
            @compat (; f, x, tang) = deepcopy(scen)
            ex = $prep_op(f, ba, x, tang)
            JET.@test_opt $op(f, ex, ba, x, tang)
            JET.@test_call $op(f, ex, ba, x, tang)
            JET.@test_opt $val_and_op(f, ex, ba, x, tang)
            JET.@test_call $val_and_op(f, ex, ba, x, tang)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, tang, res1, res2) = deepcopy(scen)
            ex = $prep_op(f, ba, x, tang)
            JET.@test_opt $op!(f, res1, ex, ba, x, tang)
            JET.@test_call $op!(f, res1, ex, ba, x, tang)
            JET.@test_opt $val_and_op!(f, res1, ex, ba, x, tang)
            JET.@test_call $val_and_op!(f, res1, ex, ba, x, tang)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S2out)
            @compat (; f, x, y, tang) = deepcopy(scen)
            ex = $prep_op(f, y, ba, x, tang)
            JET.@test_opt $op(f, y, ex, ba, x, tang)
            JET.@test_call $op(f, y, ex, ba, x, tang)
            JET.@test_opt $val_and_op(f, y, ex, ba, x, tang)
            JET.@test_call $val_and_op(f, y, ex, ba, x, tang)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S2in)
            @compat (; f, x, y, tang, res1) = deepcopy(scen)
            ex = $prep_op(f, y, ba, x, tang)
            JET.@test_opt $op!(f, y, res1, ex, ba, x, tang)
            JET.@test_call $op!(f, y, res1, ex, ba, x, tang)
            JET.@test_opt $val_and_op!(f, y, res1, ex, ba, x, tang)
            JET.@test_call $val_and_op!(f, y, res1, ex, ba, x, tang)
        end

    elseif op in [:hvp]
        @eval function test_jet(ba::AbstractADType, scen::$S1out)
            @compat (; f, x, tang) = deepcopy(scen)
            ex = $prep_op(f, ba, x, tang)
            JET.@test_opt $op(f, ex, ba, x, tang)
            JET.@test_call $op(f, ex, ba, x, tang)
        end

        @eval function test_jet(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, tang, res1, res2) = deepcopy(scen)
            ex = $prep_op(f, ba, x, tang)
            JET.@test_opt $op!(f, res2, ex, ba, x, tang)
            JET.@test_call $op!(f, res2, ex, ba, x, tang)
        end
    end
end
