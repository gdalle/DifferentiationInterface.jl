for op in ALL_OPS
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
            ba::AbstractADType, scen::$S1out; subset::Symbol, ignored_modules
        )
            (; f, x, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules $prep_op(f, ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules $op(f, ba, x, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $val_and_op(
                f, ba, x, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op(f, prep, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $val_and_op(
                    f, prep, ba, x, contexts...
                )
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType, scen::$S1in; subset::Symbol, ignored_modules
        )
            (; f, x, res1, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules $prep_op(f, ba, x, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $op!(
                f, res1, ba, x, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $val_and_op!(
                f, res1, ba, x, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op!(
                    f, res1, prep, ba, x, contexts...
                )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $val_and_op!(
                    f, res1, prep, ba, x, contexts...
                )
            return nothing
        end

        op == :gradient && continue

        @eval function test_jet(
            ba::AbstractADType, scen::$S2out; subset::Symbol, ignored_modules
        )
            (; f, x, y, contexts) = deepcopy(scen)
            prep = $prep_op(f, y, ba, x, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $prep_op(
                f, y, ba, x, contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules $op(f, y, ba, x, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $val_and_op(
                f, y, ba, x, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op(
                    f, y, prep, ba, x, contexts...
                )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $val_and_op(
                    f, y, prep, ba, x, contexts...
                )
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType, scen::$S2in; subset::Symbol, ignored_modules
        )
            (; f, x, y, res1, contexts) = deepcopy(scen)
            prep = $prep_op(f, y, ba, x, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $prep_op(
                f, y, ba, x, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $op!(
                f, y, res1, ba, x, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $val_and_op!(
                f, y, res1, ba, x, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op!(
                    f, y, res1, prep, ba, x, contexts...
                )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $val_and_op!(
                    f, y, res1, prep, ba, x, contexts...
                )
            return nothing
        end

    elseif op in [:second_derivative, :hessian]
        @eval function test_jet(
            ba::AbstractADType, scen::$S1out; subset::Symbol, ignored_modules
        )
            (; f, x, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules $prep_op(f, ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules $op(f, ba, x, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $val_and_op(
                f, ba, x, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op(f, prep, ba, x, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $val_and_op(
                    f, prep, ba, x, contexts...
                )
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType, scen::$S1in; subset::Symbol, ignored_modules
        )
            (; f, x, res1, res2, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, contexts...)
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules $prep_op(f, ba, x, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $op!(
                f, res2, ba, x, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $val_and_op!(
                f, res1, res2, ba, x, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op!(
                    f, res2, prep, ba, x, contexts...
                )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $val_and_op!(
                    f, res1, res2, prep, ba, x, contexts...
                )
            return nothing
        end

    elseif op in [:pushforward, :pullback]
        @eval function test_jet(
            ba::AbstractADType, scen::$S1out; subset::Symbol, ignored_modules
        )
            (; f, x, tang, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, tang, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $prep_op(
                f, ba, x, tang, contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules $op(f, ba, x, tang, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $val_and_op(
                f, ba, x, tang, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op(
                    f, prep, ba, x, tang, contexts...
                )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $val_and_op(
                    f, prep, ba, x, tang, contexts...
                )
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType, scen::$S1in; subset::Symbol, ignored_modules
        )
            (; f, x, tang, res1, res2, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, tang, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $prep_op(
                f, ba, x, tang, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $op!(
                f, res1, ba, x, tang, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $val_and_op!(
                f, res1, ba, x, tang, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op!(
                    f, res1, prep, ba, x, tang, contexts...
                )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $val_and_op!(
                    f, res1, prep, ba, x, tang, contexts...
                )
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType, scen::$S2out; subset::Symbol, ignored_modules
        )
            (; f, x, y, tang, contexts) = deepcopy(scen)
            prep = $prep_op(f, y, ba, x, tang, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $prep_op(
                f, y, ba, x, tang, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $op(
                f, y, ba, x, tang, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $val_and_op(
                f, y, ba, x, tang, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op(
                    f, y, prep, ba, x, tang, contexts...
                )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $val_and_op(
                    f, y, prep, ba, x, tang, contexts...
                )
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType, scen::$S2in; subset::Symbol, ignored_modules
        )
            (; f, x, y, tang, res1, contexts) = deepcopy(scen)
            prep = $prep_op(f, y, ba, x, tang, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $prep_op(
                f, y, ba, x, tang, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $op!(
                f, y, res1, ba, x, tang, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $val_and_op!(
                f, y, res1, ba, x, tang, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op!(
                    f, y, res1, prep, ba, x, tang, contexts...
                )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $val_and_op!(
                    f, y, res1, prep, ba, x, tang, contexts...
                )
            return nothing
        end

    elseif op in [:hvp]
        @eval function test_jet(
            ba::AbstractADType, scen::$S1out; subset::Symbol, ignored_modules
        )
            (; f, x, tang, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, tang, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $prep_op(
                f, ba, x, tang, contexts...
            )
            (subset == :full) &&
                @test_opt ignored_modules = ignored_modules $op(f, ba, x, tang, contexts...)
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op(
                    f, prep, ba, x, tang, contexts...
                )
            return nothing
        end

        @eval function test_jet(
            ba::AbstractADType, scen::$S1in; subset::Symbol, ignored_modules
        )
            (; f, x, tang, res1, res2, contexts) = deepcopy(scen)
            prep = $prep_op(f, ba, x, tang, contexts...)
            (subset == :full) && @test_opt ignored_modules = ignored_modules $prep_op(
                f, ba, x, tang, contexts...
            )
            (subset == :full) && @test_opt ignored_modules = ignored_modules $op!(
                f, res2, ba, x, tang, contexts...
            )
            (subset in (:prepared, :full)) &&
                @test_opt ignored_modules = ignored_modules $op!(
                    f, res2, prep, ba, x, tang, contexts...
                )
            return nothing
        end
    end
end
