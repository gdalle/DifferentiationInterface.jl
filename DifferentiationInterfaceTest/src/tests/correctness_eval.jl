function test_scen_intact(new_scen, scen; isequal)
    @testset "Scenario intact" begin
        for n in fieldnames(typeof(scen))
            n == :f && continue
            @test isequal(getfield(new_scen, n), getfield(scen, n))
        end
    end
end

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
    prep_op_same = Symbol("prepare_", op, "_same_point")

    E = if op == :derivative
        DerivativeExtras
    elseif op == :gradient
        GradientExtras
    elseif op == :hessian
        HessianExtras
    elseif op == :hvp
        HVPExtras
    elseif op == :jacobian
        JacobianExtras
    elseif op == :pullback
        PullbackExtras
    elseif op == :pushforward
        PushforwardExtras
    elseif op == :second_derivative
        SecondDerivativeExtras
    end

    S1out = Scenario{op,1,:outofplace}
    S1in = Scenario{op,1,:inplace}
    S2out = Scenario{op,2,:outofplace}
    S2in = Scenario{op,2,:inplace}

    if op in [:derivative, :gradient, :jacobian]
        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1out;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, res1) = new_scen = deepcopy(scen)
            xrand = mycopy_random(x)
            extup_cands_val, extup_cands_noval = map(1:2) do _
                [(), ($prep_op(f, ba, xrand),)]
            end
            for (extup_val, extup_noval) in zip(extup_cands_val, extup_cands_noval)
                y_out1_val, res1_out1_val = $val_and_op(f, extup_val..., ba, x)
                y_out2_val, res1_out2_val = $val_and_op(f, extup_val..., ba, x)
                res1_out1_noval = $op(f, extup_noval..., ba, x)
                res1_out2_noval = $op(f, extup_noval..., ba, x)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1in;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, res1) = new_scen = deepcopy(scen)
            xrand = mycopy_random(x)
            extup_cands_val, extup_cands_noval = map(1:2) do _
                [(), ($prep_op(f, ba, xrand),)]
            end
            for (extup_val, extup_noval) in zip(extup_cands_val, extup_cands_noval)
                res1_in1_val = mysimilar(res1)
                res1_in2_val = mysimilar(res1)
                res1_in1_noval = mysimilar(res1)
                res1_in2_noval = mysimilar(res1)
                y_out1_val, res1_out1_val = $val_and_op!(
                    f, res1_in1_val, extup_val..., ba, x
                )
                y_out2_val, res1_out2_val = $val_and_op!(
                    f, res1_in2_val, extup_val..., ba, x
                )
                res1_out1_noval = $op!(f, res1_in1_noval, extup_noval..., ba, x)
                res1_out2_noval = $op!(f, res1_in2_noval, extup_noval..., ba, x)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_in1_val ≈ scen.res1
                    @test res1_in2_val ≈ scen.res1
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_in1_noval ≈ scen.res1
                    @test res1_in2_noval ≈ scen.res1
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

        op == :gradient && continue

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S2out;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, res1) = new_scen = deepcopy(scen)
            xrand, yrand = mycopy_random(x), mycopy_random(y)
            extup_cands_val, extup_cands_noval = map(1:2) do _
                [(), ($prep_op(f, yrand, ba, xrand),)]
            end
            for (extup_val, extup_noval) in zip(extup_cands_val, extup_cands_noval)
                y_in1_val = mysimilar(y)
                y_in2_val = mysimilar(y)
                y_in1_noval = mysimilar(y)
                y_in2_noval = mysimilar(y)
                y_out1_val, res1_out1_val = $val_and_op(f, y_in1_val, extup_val..., ba, x)
                y_out2_val, res1_out2_val = $val_and_op(f, y_in2_val, extup_val..., ba, x)
                res1_out1_noval = $op(f, y_in1_noval, extup_noval..., ba, x)
                res1_out2_noval = $op(f, y_in2_noval, extup_noval..., ba, x)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test y_in1_val ≈ scen.y
                    @test y_in2_val ≈ scen.y
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S2in;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, res1) = new_scen = deepcopy(scen)
            xrand, yrand = mycopy_random(x), mycopy_random(y)
            extup_cands_val, extup_cands_noval = map(1:2) do _
                [(), ($prep_op(f, yrand, ba, xrand),)]
            end
            for (extup_val, extup_noval) in zip(extup_cands_val, extup_cands_noval)
                y_in1_val, res1_in1_val = mysimilar(y), mysimilar(res1)
                y_in2_val, res1_in2_val = mysimilar(y), mysimilar(res1)
                y_in1_noval, res1_in1_noval = mysimilar(y), mysimilar(res1)
                y_in2_noval, res1_in2_noval = mysimilar(y), mysimilar(res1)
                y_out1_val, res1_out1_val = $val_and_op!(
                    f, y_in1_val, res1_in1_val, extup_val..., ba, x
                )
                y_out2_val, res1_out2_val = $val_and_op!(
                    f, y_in2_val, res1_in2_val, extup_val..., ba, x
                )
                res1_out1_noval = $op!(
                    f, y_in1_noval, res1_in1_noval, extup_noval..., ba, x
                )
                res1_out2_noval = $op!(
                    f, y_in2_noval, res1_in2_noval, extup_noval..., ba, x
                )
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test y_in1_val ≈ scen.y
                    @test y_in2_val ≈ scen.y
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_in1_val ≈ scen.res1
                    @test res1_in2_val ≈ scen.res1
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_in1_noval ≈ scen.res1
                    @test res1_in2_noval ≈ scen.res1
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

    elseif op in [:second_derivative, :hessian]
        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1out;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, res1, res2) = new_scen = deepcopy(scen)
            xrand = mycopy_random(x)
            extup_cands_val, extup_cands_noval = map(1:2) do _
                [(), ($prep_op(f, ba, xrand),)]
            end
            for (extup_val, extup_noval) in zip(extup_cands_val, extup_cands_noval)
                y_out1_val, res1_out1_val, res2_out1_val = $val_and_op(
                    f, extup_val..., ba, x
                )
                y_out2_val, res1_out2_val, res2_out2_val = $val_and_op(
                    f, extup_val..., ba, x
                )
                res2_out1_noval = $op(f, extup_noval..., ba, x)
                res2_out2_noval = $op(f, extup_noval..., ba, x)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res2_out1_val ≈ scen.res2
                    @test res2_out2_val ≈ scen.res2
                    @test res2_out1_noval ≈ scen.res2
                    @test res2_out2_noval ≈ scen.res2
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1in;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, res1, res2) = new_scen = deepcopy(scen)
            xrand = mycopy_random(x)
            extup_cands_val, extup_cands_noval = map(1:2) do _
                [(), ($prep_op(f, ba, xrand),)]
            end
            for (extup_val, extup_noval) in zip(extup_cands_val, extup_cands_noval)
                res1_in1_val, res2_in1_val = mysimilar(res1), mysimilar(res2)
                res1_in2_val, res2_in2_val = mysimilar(res1), mysimilar(res2)
                res2_in1_noval = mysimilar(res2)
                res2_in2_noval = mysimilar(res2)
                y_out1_val, res1_out1_val, res2_out1_val = $val_and_op!(
                    f, res1_in1_val, res2_in1_val, extup_val..., ba, x
                )
                y_out2_val, res1_out2_val, res2_out2_val = $val_and_op!(
                    f, res1_in2_val, res2_in2_val, extup_val..., ba, x
                )
                res2_out1_noval = $op!(f, res2_in1_noval, extup_noval..., ba, x)
                res2_out2_noval = $op!(f, res2_in2_noval, extup_noval..., ba, x)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_in1_val ≈ scen.res1
                    @test res1_in2_val ≈ scen.res1
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res2_in1_val ≈ scen.res2
                    @test res2_in2_val ≈ scen.res2
                    @test res2_out1_val ≈ scen.res2
                    @test res2_out2_val ≈ scen.res2
                    @test res2_in1_noval ≈ scen.res2
                    @test res2_in2_noval ≈ scen.res2
                    @test res2_out1_noval ≈ scen.res2
                    @test res2_out2_noval ≈ scen.res2
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

    elseif op in [:pushforward, :pullback]
        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1out;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, seed, res1) = new_scen = deepcopy(scen)
            xrand, seedrand = mycopy_random(x), mycopy_random(seed)
            extup_cands_val, extup_cands_noval = map(1:2) do _
                [(), ($prep_op(f, ba, xrand, seedrand),), ($prep_op_same(f, ba, x, seedrand),)]
            end
            for (extup_val, extup_noval) in zip(extup_cands_val, extup_cands_noval)
                y_out1_val, res1_out1_val = $val_and_op(f, extup_val..., ba, x, seed)
                y_out2_val, res1_out2_val = $val_and_op(f, extup_val..., ba, x, seed)
                res1_out1_noval = $op(f, extup_noval..., ba, x, seed)
                res1_out2_noval = $op(f, extup_noval..., ba, x, seed)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1in;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, seed, res1) = new_scen = deepcopy(scen)
            xrand, seedrand = mycopy_random(x), mycopy_random(seed)
            extup_cands_val, extup_cands_noval = map(1:2) do _
                [(), ($prep_op(f, ba, xrand, seedrand),), ($prep_op_same(f, ba, x, seedrand),)]
            end
            for (extup_val, extup_noval) in zip(extup_cands_val, extup_cands_noval)
                res1_in1_val = mysimilar(res1)
                res1_in2_val = mysimilar(res1)
                res1_in1_noval = mysimilar(res1)
                res1_in2_noval = mysimilar(res1)
                y_out1_val, res1_out1_val = $val_and_op!(
                    f, res1_in1_val, extup_val..., ba, x, seed
                )
                y_out2_val, res1_out2_val = $val_and_op!(
                    f, res1_in2_val, extup_val..., ba, x, seed
                )
                res1_out1_noval = $op!(f, res1_in1_noval, extup_noval..., ba, x, seed)
                res1_out2_noval = $op!(f, res1_in2_noval, extup_noval..., ba, x, seed)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_in1_val ≈ scen.res1
                    @test res1_in2_val ≈ scen.res1
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_in1_noval ≈ scen.res1
                    @test res1_in2_noval ≈ scen.res1
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S2out;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, seed, res1) = new_scen = deepcopy(scen)
            xrand, yrand, seedrand = mycopy_random(x), mycopy_random(y), mycopy_random(seed)
            extup_cands_val, extup_cands_noval = map(1:2) do _
                [
                    (),
                    ($prep_op(f, yrand, ba, xrand, seedrand),),
                    ($prep_op_same(f, yrand, ba, x, seedrand),),
                ]
            end
            for (extup_val, extup_noval) in zip(extup_cands_val, extup_cands_noval)
                y_in1_val = mysimilar(y)
                y_in2_val = mysimilar(y)
                y_in1_noval = mysimilar(y)
                y_in2_noval = mysimilar(y)
                y_out1_val, res1_out1_val = $val_and_op(
                    f, y_in1_val, extup_val..., ba, x, seed
                )
                y_out2_val, res1_out2_val = $val_and_op(
                    f, y_in2_val, extup_val..., ba, x, seed
                )
                res1_out1_noval = $op(f, y_in1_noval, extup_noval..., ba, x, seed)
                res1_out2_noval = $op(f, y_in2_noval, extup_noval..., ba, x, seed)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test y_in1_val ≈ scen.y
                    @test y_in2_val ≈ scen.y
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S2in;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, seed, res1) = new_scen = deepcopy(scen)
            xrand, yrand, seedrand = mycopy_random(x), mycopy_random(y), mycopy_random(seed)
            extup_cands_val, extup_cands_noval = map(1:2) do _
                [
                    (),
                    ($prep_op(f, yrand, ba, xrand, seedrand),),
                    ($prep_op_same(f, yrand, ba, x, seedrand),),
                ]
            end
            for (extup_val, extup_noval) in zip(extup_cands_val, extup_cands_noval)
                y_in1_val, res1_in1_val = mysimilar(y), mysimilar(res1)
                y_in2_val, res1_in2_val = mysimilar(y), mysimilar(res1)
                y_in1_noval, res1_in1_noval = mysimilar(y), mysimilar(res1)
                y_in2_noval, res1_in2_noval = mysimilar(y), mysimilar(res1)
                y_out1_val, res1_out1_val = $val_and_op!(
                    f, y_in1_val, res1_in1_val, extup_val..., ba, x, seed
                )
                y_out2_val, res1_out2_val = $val_and_op!(
                    f, y_in2_val, res1_in2_val, extup_val..., ba, x, seed
                )
                res1_out1_noval = $op!(
                    f, y_in1_noval, res1_in1_noval, extup_noval..., ba, x, seed
                )
                res1_out2_noval = $op!(
                    f, y_in2_noval, res1_in2_noval, extup_noval..., ba, x, seed
                )
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test y_in1_val ≈ scen.y
                    @test y_in2_val ≈ scen.y
                    @test y_out1_val ≈ scen.y
                    @test y_out2_val ≈ scen.y
                    @test res1_in1_val ≈ scen.res1
                    @test res1_in2_val ≈ scen.res1
                    @test res1_out1_val ≈ scen.res1
                    @test res1_out2_val ≈ scen.res1
                    @test res1_in1_noval ≈ scen.res1
                    @test res1_in2_noval ≈ scen.res1
                    @test res1_out1_noval ≈ scen.res1
                    @test res1_out2_noval ≈ scen.res1
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

    elseif op in [:hvp]
        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1out;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, seed, res2) = new_scen = deepcopy(scen)
            xrand, seedrand = mycopy_random(x), mycopy_random(seed)
            extup_cands_noval = [
                (),
                ($prep_op(f, ba, xrand, seedrand),),
                ($prep_op_same(f, ba, x, seedrand),),
            ]
            for extup_noval in extup_cands_noval
                res2_out1_noval = $op(f, extup_noval..., ba, x, seed)
                res2_out2_noval = $op(f, extup_noval..., ba, x, seed)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test res2_out1_noval ≈ scen.res2
                    @test res2_out2_noval ≈ scen.res2
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end

        @eval function test_correctness(
            ba::AbstractADType,
            scen::$S1in;
            isequal::Function,
            isapprox::Function,
            atol::Real,
            rtol::Real,
        )
            @compat (; f, x, y, seed, res2) = new_scen = deepcopy(scen)
            xrand, seedrand = mycopy_random(x), mycopy_random(seed)
            extup_cands_noval = [
                (),
                ($prep_op(f, ba, xrand, seedrand),),
                ($prep_op_same(f, ba, x, seedrand),),
            ]
            for extup_noval in extup_cands_noval
                res2_in1_noval = mysimilar(res2)
                res2_in2_noval = mysimilar(res2)
                res2_out1_noval = $op!(f, res2_in1_noval, extup_noval..., ba, x, seed)
                res2_out2_noval = $op!(f, res2_in2_noval, extup_noval..., ba, x, seed)
                let (≈)(x, y) = isapprox(x, y; atol, rtol)
                    @test isempty(extup_noval) || only(extup_noval) isa $E
                    @test res2_in1_noval ≈ scen.res2
                    @test res2_in2_noval ≈ scen.res2
                    @test res2_out1_noval ≈ scen.res2
                    @test res2_out2_noval ≈ scen.res2
                end
            end
            test_scen_intact(new_scen, scen; isequal)
            return nothing
        end
    end
end
