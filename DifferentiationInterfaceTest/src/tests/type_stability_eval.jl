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
        @eval function test_jet(
            ba::AbstractADType,
            scen::$S1out;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, contexts...)
            preparation && JET.@test_opt $prep_op(f, ba, x, contexts...)
            prepared_op && JET.@test_opt $op(f, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $op(f, prep, ba, x, contexts...)
            prepared_op && JET.@test_opt $val_and_op(f, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $val_and_op(f, prep, ba, x, contexts...)
            unprepared_op && JET.@test_opt $op(f, ba, x, contexts...)
            unprepared_op && JET.@test_call $op(f, ba, x, contexts...)
            unprepared_op && JET.@test_opt $val_and_op(f, ba, x, contexts...)
            unprepared_op && JET.@test_call $val_and_op(f, ba, x, contexts...)
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType,
            scen::$S1in;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, res1, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, contexts...)
            preparation && JET.@test_opt $prep_op(f, ba, x, contexts...)
            prepared_op && JET.@test_opt $op!(f, res1, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $op!(f, res1, prep, ba, x, contexts...)
            prepared_op && JET.@test_opt $val_and_op!(f, res1, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $val_and_op!(f, res1, prep, ba, x, contexts...)
            unprepared_op && JET.@test_opt $op!(f, res1, ba, x, contexts...)
            unprepared_op && JET.@test_call $op!(f, res1, ba, x, contexts...)
            unprepared_op && JET.@test_opt $val_and_op!(f, res1, ba, x, contexts...)
            unprepared_op && JET.@test_call $val_and_op!(f, res1, ba, x, contexts...)
            return nothing
        end

        op == :gradient && continue

        @eval function test_jet(
            ba::AbstractADType,
            scen::$S2out;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, y, contexts) = deepcopy(scen)
            prep = $prep_op(f, y, ba, x, contexts...)
            preparation && JET.@test_opt $prep_op(f, y, ba, x, contexts...)
            prepared_op && JET.@test_opt $op(f, y, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $op(f, y, prep, ba, x, contexts...)
            prepared_op && JET.@test_opt $val_and_op(f, y, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $val_and_op(f, y, prep, ba, x, contexts...)
            unprepared_op && JET.@test_opt $op(f, y, ba, x, contexts...)
            unprepared_op && JET.@test_call $op(f, y, ba, x, contexts...)
            unprepared_op && JET.@test_opt $val_and_op(f, y, ba, x, contexts...)
            unprepared_op && JET.@test_call $val_and_op(f, y, ba, x, contexts...)
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType,
            scen::$S2in;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, y, res1, contexts) = deepcopy(scen)
            prep = $prep_op(f, y, ba, x, contexts...)
            preparation && JET.@test_opt $prep_op(f, y, ba, x, contexts...)
            prepared_op && JET.@test_opt $op!(f, y, res1, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $op!(f, y, res1, prep, ba, x, contexts...)
            prepared_op && JET.@test_opt $val_and_op!(f, y, res1, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $val_and_op!(f, y, res1, prep, ba, x, contexts...)
            unprepared_op && JET.@test_opt $op!(f, y, res1, ba, x, contexts...)
            unprepared_op && JET.@test_call $op!(f, y, res1, ba, x, contexts...)
            unprepared_op && JET.@test_opt $val_and_op!(f, y, res1, ba, x, contexts...)
            unprepared_op && JET.@test_call $val_and_op!(f, y, res1, ba, x, contexts...)
            return nothing
        end

    elseif op in [:second_derivative, :hessian]
        @eval function test_jet(
            ba::AbstractADType,
            scen::$S1out;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, contexts...)
            preparation && JET.@test_opt $prep_op(f, ba, x, contexts...)
            prepared_op && JET.@test_opt $op(f, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $op(f, prep, ba, x, contexts...)
            prepared_op && JET.@test_opt $val_and_op(f, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $val_and_op(f, prep, ba, x, contexts...)
            unprepared_op && JET.@test_opt $op(f, ba, x, contexts...)
            unprepared_op && JET.@test_call $op(f, ba, x, contexts...)
            unprepared_op && JET.@test_opt $val_and_op(f, ba, x, contexts...)
            unprepared_op && JET.@test_call $val_and_op(f, ba, x, contexts...)
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType,
            scen::$S1in;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, res1, res2, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, contexts...)
            preparation && JET.@test_opt $prep_op(f, ba, x, contexts...)
            prepared_op && JET.@test_opt $op!(f, res2, prep, ba, x, contexts...)
            prepared_op && JET.@test_call $op!(f, res2, prep, ba, x, contexts...)
            prepared_op &&
                JET.@test_opt $val_and_op!(f, res1, res2, prep, ba, x, contexts...)
            prepared_op &&
                JET.@test_call $val_and_op!(f, res1, res2, prep, ba, x, contexts...)
            unprepared_op && JET.@test_opt $op!(f, res2, ba, x, contexts...)
            unprepared_op && JET.@test_call $op!(f, res2, ba, x, contexts...)
            unprepared_op && JET.@test_opt $val_and_op!(f, res1, res2, ba, x, contexts...)
            unprepared_op && JET.@test_call $val_and_op!(f, res1, res2, ba, x, contexts...)
            return nothing
        end

    elseif op in [:pushforward, :pullback]
        @eval function test_jet(
            ba::AbstractADType,
            scen::$S1out;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, tang, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, tang, contexts...)
            preparation && JET.@test_opt $prep_op(f, ba, x, tang, contexts...)
            prepared_op && JET.@test_opt $op(f, prep, ba, x, tang, contexts...)
            prepared_op && JET.@test_call $op(f, prep, ba, x, tang, contexts...)
            prepared_op && JET.@test_opt $val_and_op(f, prep, ba, x, tang, contexts...)
            prepared_op && JET.@test_call $val_and_op(f, prep, ba, x, tang, contexts...)
            unprepared_op && JET.@test_opt $op(f, ba, x, tang, contexts...)
            unprepared_op && JET.@test_call $op(f, ba, x, tang, contexts...)
            unprepared_op && JET.@test_opt $val_and_op(f, ba, x, tang, contexts...)
            unprepared_op && JET.@test_call $val_and_op(f, ba, x, tang, contexts...)
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType,
            scen::$S1in;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, tang, res1, res2, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, tang, contexts...)
            preparation && JET.@test_opt $prep_op(f, ba, x, tang, contexts...)
            prepared_op && JET.@test_opt $op!(f, res1, prep, ba, x, tang, contexts...)
            prepared_op && JET.@test_call $op!(f, res1, prep, ba, x, tang, contexts...)
            prepared_op &&
                JET.@test_opt $val_and_op!(f, res1, prep, ba, x, tang, contexts...)
            prepared_op &&
                JET.@test_call $val_and_op!(f, res1, prep, ba, x, tang, contexts...)
            unprepared_op && JET.@test_opt $op!(f, res1, ba, x, tang, contexts...)
            unprepared_op && JET.@test_call $op!(f, res1, ba, x, tang, contexts...)
            unprepared_op && JET.@test_opt $val_and_op!(f, res1, ba, x, tang, contexts...)
            unprepared_op && JET.@test_call $val_and_op!(f, res1, ba, x, tang, contexts...)
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType,
            scen::$S2out;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, y, tang, contexts) = deepcopy(scen)
            prep = $prep_op(f, y, ba, x, tang, contexts...)
            preparation && JET.@test_opt $prep_op(f, y, ba, x, tang, contexts...)
            prepared_op && JET.@test_opt $op(f, y, prep, ba, x, tang, contexts...)
            prepared_op && JET.@test_call $op(f, y, prep, ba, x, tang, contexts...)
            prepared_op && JET.@test_opt $val_and_op(f, y, prep, ba, x, tang, contexts...)
            prepared_op && JET.@test_call $val_and_op(f, y, prep, ba, x, tang, contexts...)
            unprepared_op && JET.@test_opt $op(f, y, ba, x, tang, contexts...)
            unprepared_op && JET.@test_call $op(f, y, ba, x, tang, contexts...)
            unprepared_op && JET.@test_opt $val_and_op(f, y, ba, x, tang, contexts...)
            unprepared_op && JET.@test_call $val_and_op(f, y, ba, x, tang, contexts...)
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType,
            scen::$S2in;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, y, tang, res1, contexts) = deepcopy(scen)
            prep = $prep_op(f, y, ba, x, tang, contexts...)
            preparation && JET.@test_opt $prep_op(f, y, ba, x, tang, contexts...)
            prepared_op && JET.@test_opt $op!(f, y, res1, prep, ba, x, tang, contexts...)
            prepared_op && JET.@test_call $op!(f, y, res1, prep, ba, x, tang, contexts...)
            prepared_op &&
                JET.@test_opt $val_and_op!(f, y, res1, prep, ba, x, tang, contexts...)
            prepared_op &&
                JET.@test_call $val_and_op!(f, y, res1, prep, ba, x, tang, contexts...)
            unprepared_op && JET.@test_opt $op!(f, y, res1, ba, x, tang, contexts...)
            unprepared_op && JET.@test_call $op!(f, y, res1, ba, x, tang, contexts...)
            unprepared_op &&
                JET.@test_opt $val_and_op!(f, y, res1, ba, x, tang, contexts...)
            unprepared_op &&
                JET.@test_call $val_and_op!(f, y, res1, ba, x, tang, contexts...)
            return nothing
        end

    elseif op in [:hvp]
        @eval function test_jet(
            ba::AbstractADType,
            scen::$S1out;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, tang, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, tang, contexts...)
            preparation && JET.@test_opt $prep_op(f, ba, x, tang, contexts...)
            prepared_op && JET.@test_opt $op(f, prep, ba, x, tang, contexts...)
            prepared_op && JET.@test_call $op(f, prep, ba, x, tang, contexts...)
            unprepared_op && JET.@test_opt $op(f, ba, x, tang, contexts...)
            unprepared_op && JET.@test_call $op(f, ba, x, tang, contexts...)
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType,
            scen::$S1in;
            preparation::Bool,
            prepared_op::Bool,
            unprepared_op::Bool,
        )
            (; f, x, tang, res1, res2, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, tang, contexts...)
            preparation && JET.@test_opt $prep_op(f, ba, x, tang, contexts...)
            prepared_op && JET.@test_opt $op!(f, res2, prep, ba, x, tang, contexts...)
            prepared_op && JET.@test_call $op!(f, res2, prep, ba, x, tang, contexts...)
            unprepared_op && JET.@test_opt $op!(f, res2, ba, x, tang, contexts...)
            unprepared_op && JET.@test_call $op!(f, res2, ba, x, tang, contexts...)
            return nothing
        end
    end
end
