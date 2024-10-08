
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

    @eval function run_benchmark!(
        data::Vector{DifferentiationBenchmarkDataRow},
        backend::AbstractADType,
        scenario::Union{$S1out,$S1in,$S2out,$S2in};
        logging::Bool,
    )
        (; bench0, bench1, bench2, calls0, calls1, calls2) = try
            run_benchmark_aux(backend, scenario)
        catch exception
            logging && @warn "Error during benchmarking" backend scenario exception
            bench0, bench1, bench2 = failed_benchs(3)
            calls0, calls1, calls2 = -1, -1, -1
            (; bench0, bench1, bench2, calls0, calls1, calls2)
        end
        record!(data, backend, scenario, $prep_op, bench0, calls0)
        if scenario isa Union{$S1out,$S2out}
            record!(data, backend, scenario, $(string(val_and_op)), bench1, calls1)
            record!(data, backend, scenario, $(string(op)), bench2, calls2)
        elseif scenario isa Union{$S1in,$S2in}
            record!(data, backend, scenario, $(string(val_and_op!)), bench1, calls1)
            record!(data, backend, scenario, $(string(op!)), bench2, calls2)
        end
        return nothing
    end

    if op in [:derivative, :gradient, :jacobian]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            (; f, x, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, ba, x, contexts...)
            bench0 = @be $prep_op(f, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be prep $val_and_op(f, _, ba, x, contexts...) evals = 1
            bench2 = @be prep $op(f, _, ba, x, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op(cc, prep, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op(cc, prep, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            (; f, x, res1, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, ba, x, contexts...)
            bench0 = @be $prep_op(f, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be (res1, prep) $val_and_op!(f, _[1], _[2], ba, x, contexts...) evals =
                1
            bench2 = @be (res1, prep) $op!(f, _[1], _[2], ba, x, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, res1, prep, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op!(cc, res1, prep, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        op == :gradient && continue

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2out)
            (; f, x, y, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, y, ba, x, contexts...)
            bench0 = @be $prep_op(f, y, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be (y, prep) $val_and_op(f, _[1], _[2], ba, x, contexts...) evals = 1
            bench2 = @be (y, prep) $op(f, _[1], _[2], ba, x, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, y, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op(cc, y, prep, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op(cc, y, prep, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2in)
            (; f, x, y, res1, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, y, ba, x, contexts...)
            bench0 = @be $prep_op(f, y, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be (y, res1, prep) $val_and_op!(
                f, _[1], _[2], _[3], ba, x, contexts...
            ) evals = 1
            bench2 = @be (y, res1, prep) $op!(f, _[1], _[2], _[3], ba, x, contexts...) evals =
                1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, y, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, y, res1, prep, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op!(cc, y, res1, prep, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

    elseif op in [:hessian, :second_derivative]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            (; f, x, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, ba, x, contexts...)
            bench0 = @be $prep_op(f, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be prep $val_and_op(f, _, ba, x, contexts...) evals = 1
            bench2 = @be prep $op(f, _, ba, x, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op(cc, prep, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op(cc, prep, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            (; f, x, res1, res2, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, ba, x, contexts...)
            bench0 = @be $prep_op(f, ba, x, contexts...) samples = 1 evals = 1
            bench1 = @be (res1, res2, prep) $val_and_op!(
                f, _[1], _[2], _[3], ba, x, contexts...
            ) evals = 1
            bench2 = @be (res2, prep) $op!(f, _[1], _[2], ba, x, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, x, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, res1, res2, prep, ba, x, contexts...)
            calls1 = reset_count!(cc)
            $op!(cc, res2, prep, ba, x, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

    elseif op in [:pushforward, :pullback]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            (; f, x, tang, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be prep $val_and_op(f, _, ba, x, tang, contexts...) evals = 1
            bench2 = @be prep $op(f, _, ba, x, tang, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op(cc, prep, ba, x, tang, contexts...)
            calls1 = reset_count!(cc)
            $op(cc, prep, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            (; f, x, tang, res1, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be (res1, prep) $val_and_op!(f, _[1], _[2], ba, x, tang, contexts...) evals =
                1
            bench2 = @be (res1, prep) $op!(f, _[1], _[2], ba, x, tang, contexts...) evals =
                1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, res1, prep, ba, x, tang, contexts...)
            calls1 = reset_count!(cc)
            $op!(cc, res1, prep, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2out)
            (; f, x, y, tang, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, y, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, y, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be (y, prep) $val_and_op(f, _[1], _[2], ba, x, tang, contexts...) evals =
                1
            bench2 = @be (y, prep) $op(f, _[1], _[2], ba, x, tang, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, y, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op(cc, y, prep, ba, x, tang, contexts...)
            calls1 = reset_count!(cc)
            $op(cc, y, prep, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2in)
            (; f, x, y, tang, res1, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, y, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, y, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be (y, res1, prep) $val_and_op!(
                f, _[1], _[2], _[3], ba, x, tang, contexts...
            ) evals = 1
            bench2 = @be (y, res1, prep) $op!(f, _[1], _[2], _[3], ba, x, tang, contexts...) evals =
                1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, y, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, y, res1, prep, ba, x, tang, contexts...)
            calls1 = reset_count!(cc)
            $op!(cc, y, res1, prep, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

    elseif op in [:hvp]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            (; f, x, tang, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be +(1, 1) evals = 1  # TODO: fix
            bench2 = @be prep $op(f, _, ba, x, tang, contexts...) evals = 1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            calls1 = -1  # TODO: fix
            $op(cc, prep, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            (; f, x, tang, res2, contexts) = deepcopy(scen)
            # benchmark
            prep = $prep_op(f, ba, x, tang, contexts...)
            bench0 = @be $prep_op(f, ba, x, tang, contexts...) samples = 1 evals = 1
            bench1 = @be +(1, 1) evals = 1  # TODO: fix
            bench2 = @be (res2, prep) $op!(f, _[1], _[2], ba, x, tang, contexts...) evals =
                1
            # count
            cc = CallCounter(f)
            prep = $prep_op(cc, ba, x, tang, contexts...)
            calls0 = reset_count!(cc)
            calls1 = -1  # TODO: fix
            $op!(cc, res2, prep, ba, x, tang, contexts...)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end
    end
end
