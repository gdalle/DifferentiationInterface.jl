
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

    S1out = Scenario{op,1,:outofplace}
    S1in = Scenario{op,1,:inplace}
    S2out = Scenario{op,2,:outofplace}
    S2in = Scenario{op,2,:inplace}

    @eval function run_benchmark!(
        data::Vector{DifferentiationBenchmarkDataRow},
        backend::AbstractADType,
        scenario::Union{$S1out,$S1in,$S2out,$S2in};
        logging::Bool,
    )
        @compat (; bench0, bench1, bench2, calls0, calls1, calls2) = try
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
            @compat (; f, x) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x)
            bench0 = @be $prep_op(f, ba, x) samples = 1 evals = 1
            bench1 = @be ex $val_and_op(f, ba, x, _) evals = 1
            bench2 = @be ex $op(f, ba, x, _) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x)
            calls0 = reset_count!(cc)
            $val_and_op(cc, ba, x, ex)
            calls1 = reset_count!(cc)
            $op(cc, ba, x, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, res1) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x)
            bench0 = @be $prep_op(f, ba, x) samples = 1 evals = 1
            bench1 = @be (res1, ex) $val_and_op!(f, _[1], ba, x, _[2]) evals = 1
            bench2 = @be (res1, ex) $op!(f, _[1], ba, x, _[2]) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, res1, ba, x, ex)
            calls1 = reset_count!(cc)
            $op!(cc, res1, ba, x, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        op == :gradient && continue

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2out)
            @compat (; f, x, y) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, y, ba, x)
            bench0 = @be $prep_op(f, y, ba, x) samples = 1 evals = 1
            bench1 = @be (y, ex) $val_and_op(f, _[1], ba, x, _[2]) evals = 1
            bench2 = @be (y, ex) $op(f, _[1], ba, x, _[2]) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, y, ba, x)
            calls0 = reset_count!(cc)
            $val_and_op(cc, y, ba, x, ex)
            calls1 = reset_count!(cc)
            $op(cc, y, ba, x, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2in)
            @compat (; f, x, y, res1) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, y, ba, x)
            bench0 = @be $prep_op(f, y, ba, x) samples = 1 evals = 1
            bench1 = @be (y, res1, ex) $val_and_op!(f, _[1], _[2], ba, x, _[3]) evals = 1
            bench2 = @be (y, res1, ex) $op!(f, _[1], _[2], ba, x, _[3]) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, y, ba, x)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, y, res1, ba, x, ex)
            calls1 = reset_count!(cc)
            $op!(cc, y, res1, ba, x, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

    elseif op in [:hessian, :second_derivative]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            @compat (; f, x) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x)
            bench0 = @be $prep_op(f, ba, x) samples = 1 evals = 1
            bench1 = @be ex $val_and_op(f, ba, x, _) evals = 1
            bench2 = @be ex $op(f, ba, x, _) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x)
            calls0 = reset_count!(cc)
            $val_and_op(cc, ba, x, ex)
            calls1 = reset_count!(cc)
            $op(cc, ba, x, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, res1, res2) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x)
            bench0 = @be $prep_op(f, ba, x) samples = 1 evals = 1
            bench1 = @be (res1, res2, ex) $val_and_op!(f, _[1], _[2], ba, x, _[3]) evals = 1
            bench2 = @be (res2, ex) $op!(f, _[1], ba, x, _[2]) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, res1, res2, ba, x, ex)
            calls1 = reset_count!(cc)
            $op!(cc, res2, ba, x, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

    elseif op in [:pushforward, :pullback]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            @compat (; f, x, seed) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, seed)
            bench0 = @be $prep_op(f, ba, x, seed) samples = 1 evals = 1
            bench1 = @be ex $val_and_op(f, ba, x, seed, _) evals = 1
            bench2 = @be ex $op(f, ba, x, seed, _) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, seed)
            calls0 = reset_count!(cc)
            $val_and_op(cc, ba, x, seed, ex)
            calls1 = reset_count!(cc)
            $op(cc, ba, x, seed, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, seed, res1) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, seed)
            bench0 = @be $prep_op(f, ba, x, seed) samples = 1 evals = 1
            bench1 = @be (res1, ex) $val_and_op!(f, _[1], ba, x, seed, _[2]) evals = 1
            bench2 = @be (res1, ex) $op!(f, _[1], ba, x, seed, _[2]) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, seed)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, res1, ba, x, seed, ex)
            calls1 = reset_count!(cc)
            $op!(cc, res1, ba, x, seed, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2out)
            @compat (; f, x, y, seed) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, y, ba, x, seed)
            bench0 = @be $prep_op(f, y, ba, x, seed) samples = 1 evals = 1
            bench1 = @be (y, ex) $val_and_op(f, _[1], ba, x, seed, _[2]) evals = 1
            bench2 = @be (y, ex) $op(f, _[1], ba, x, seed, _[2]) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, y, ba, x, seed)
            calls0 = reset_count!(cc)
            $val_and_op(cc, y, ba, x, seed, ex)
            calls1 = reset_count!(cc)
            $op(cc, y, ba, x, seed, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S2in)
            @compat (; f, x, y, seed, res1) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, y, ba, x, seed)
            bench0 = @be $prep_op(f, y, ba, x, seed) samples = 1 evals = 1
            bench1 = @be (y, res1, ex) $val_and_op!(f, _[1], _[2], ba, x, seed, _[3]) evals =
                1
            bench2 = @be (y, res1, ex) $op!(f, _[1], _[2], ba, x, seed, _[3]) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, y, ba, x, seed)
            calls0 = reset_count!(cc)
            $val_and_op!(cc, y, res1, ba, x, seed, ex)
            calls1 = reset_count!(cc)
            $op!(cc, y, res1, ba, x, seed, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

    elseif op in [:hvp]
        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1out)
            @compat (; f, x, seed) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, seed)
            bench0 = @be $prep_op(f, ba, x, seed) samples = 1 evals = 1
            bench1 = @be +(1, 1) evals = 1  # TODO: fix
            bench2 = @be ex $op(f, ba, x, seed, _) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, seed)
            calls0 = reset_count!(cc)
            calls1 = -1  # TODO: fix
            $op(cc, ba, x, seed, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end

        @eval function run_benchmark_aux(ba::AbstractADType, scen::$S1in)
            @compat (; f, x, seed, res2) = deepcopy(scen)
            # benchmark
            ex = $prep_op(f, ba, x, seed)
            bench0 = @be $prep_op(f, ba, x, seed) samples = 1 evals = 1
            bench1 = @be +(1, 1) evals = 1  # TODO: fix
            bench2 = @be (res2, ex) $op!(f, _[1], ba, x, seed, _[2]) evals = 1
            # count
            cc = CallCounter(f)
            ex = $prep_op(cc, ba, x, seed)
            calls0 = reset_count!(cc)
            calls1 = -1  # TODO: fix
            $op!(cc, res2, ba, x, seed, ex)
            calls2 = reset_count!(cc)
            return (; bench0, bench1, bench2, calls0, calls1, calls2)
        end
    end
end
