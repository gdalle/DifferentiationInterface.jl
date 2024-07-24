## No overwrite

function test_scen_intact(new_scen, scen; isequal)
    @testset "Scenario intact" begin
        for n in fieldnames(typeof(scen))
            n == :f && continue
            @test isequal(getfield(new_scen, n), getfield(scen, n))
        end
    end
end

testset_name(k) = k == 1 ? "No prep" : (k == 2 ? "Different point" : "Same point")

## Pushforward

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:pushforward,1,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = if seed isa Batch
        [
            prepare_pushforward_batched(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_pushforward_batched_same_point(f, ba, x, mycopy_random(seed)),
        ]
    else
        [
            prepare_pushforward(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_pushforward_same_point(f, ba, x, mycopy_random(seed)),
        ]
    end
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        if seed isa Batch
            y1, dy1 = value_and_pushforward_batched(f, ba, x, seed, extras_tup...)
            dy2 = pushforward_batched(f, ba, x, seed, extras_tup...)
        else
            y1, dy1 = value_and_pushforward(f, ba, x, seed, extras_tup...)
            dy2 = pushforward(f, ba, x, seed, extras_tup...)
        end

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa PushforwardExtras
            end
            @testset "Primal value" begin
                @test y1 ≈ scen.y
            end
            @testset "Tangent value" begin
                @test dy1 ≈ scen.res1
                @test dy2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:pushforward,1,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = if seed isa Batch
        [
            prepare_pushforward_batched(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_pushforward_batched_same_point(f, ba, x, mycopy_random(seed)),
        ]
    else
        [
            prepare_pushforward(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_pushforward_same_point(f, ba, x, mycopy_random(seed)),
        ]
    end
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        dy1_in = mysimilar(res1)
        dy2_in = mysimilar(res1)

        if seed isa Batch
            y1, dy1 = value_and_pushforward_batched!(f, dy1_in, ba, x, seed, extras_tup...)
            dy2 = pushforward_batched!(f, dy2_in, ba, x, seed, extras_tup...)
        else
            y1, dy1 = value_and_pushforward!(f, dy1_in, ba, x, seed, extras_tup...)
            dy2 = pushforward!(f, dy2_in, ba, x, seed, extras_tup...)
        end

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa PushforwardExtras
            end
            @testset "Primal value" begin
                @test y1 ≈ scen.y
            end
            @testset "Tangent value" begin
                @test dy1_in ≈ scen.res1
                @test dy1 ≈ scen.res1
                @test dy2_in ≈ scen.res1
                @test dy2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:pushforward,2,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)
    f! = f

    extras_candidates = if seed isa Batch
        [
            prepare_pushforward_batched(
                f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(seed)
            ),
            prepare_pushforward_batched_same_point(
                f!, mysimilar(y), ba, x, mycopy_random(seed)
            ),
        ]
    else
        [
            prepare_pushforward(
                f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(seed)
            ),
            prepare_pushforward_same_point(f!, mysimilar(y), ba, x, mycopy_random(seed)),
        ]
    end
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1_in = mysimilar(y)
        y2_in = mysimilar(y)

        if seed isa Batch
            y1, dy1 = value_and_pushforward_batched(f!, y1_in, ba, x, seed, extras_tup...)
            dy2 = pushforward_batched(f!, y2_in, ba, x, seed, extras_tup...)
        else
            y1, dy1 = value_and_pushforward(f!, y1_in, ba, x, seed, extras_tup...)
            dy2 = pushforward(f!, y2_in, ba, x, seed, extras_tup...)
        end

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa PushforwardExtras
            end
            @testset "Primal value" begin
                @test y1_in ≈ scen.y
                @test y1 ≈ scen.y
            end
            @testset "Tangent value" begin
                @test dy1 ≈ scen.res1
                @test dy2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:pushforward,2,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)
    f! = f

    extras_candidates = if seed isa Batch
        [
            prepare_pushforward_batched(
                f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(seed)
            ),
            prepare_pushforward_batched_same_point(
                f!, mysimilar(y), ba, x, mycopy_random(seed)
            ),
        ]
    else
        [
            prepare_pushforward(
                f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(seed)
            ),
            prepare_pushforward_same_point(f!, mysimilar(y), ba, x, mycopy_random(seed)),
        ]
    end
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1_in, dy1_in = mysimilar(y), mysimilar(res1)
        y2_in, dy2_in = mysimilar(y), mysimilar(res1)

        if seed isa Batch
            y1, dy1 = value_and_pushforward_batched!(
                f!, y1_in, dy1_in, ba, x, seed, extras_tup...
            )
            dy2 = pushforward_batched!(f!, y2_in, dy2_in, ba, x, seed, extras_tup...)
        else
            y1, dy1 = value_and_pushforward!(f!, y1_in, dy1_in, ba, x, seed, extras_tup...)
            dy2 = pushforward!(f!, y2_in, dy2_in, ba, x, seed, extras_tup...)
        end

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa PushforwardExtras
            end
            @testset "Primal value" begin
                @test y1_in ≈ scen.y
                @test y1 ≈ scen.y
            end
            @testset "Tangent value" begin
                @test dy1_in ≈ scen.res1
                @test dy1 ≈ scen.res1
                @test dy2_in ≈ scen.res1
                @test dy2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

## Pullback

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:pullback,1,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = if seed isa Batch
        [
            prepare_pullback_batched(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_pullback_batched_same_point(f, ba, x, mycopy_random(seed)),
        ]
    else
        [
            prepare_pullback(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_pullback_same_point(f, ba, x, mycopy_random(seed)),
        ]
    end
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        if seed isa Batch
            y1, dx1 = value_and_pullback_batched(f, ba, x, seed, extras_tup...)
            dx2 = pullback_batched(f, ba, x, seed, extras_tup...)
        else
            y1, dx1 = value_and_pullback(f, ba, x, seed, extras_tup...)
            dx2 = pullback(f, ba, x, seed, extras_tup...)
        end

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa PullbackExtras
            end
            @testset "Primal value" begin
                @test y1 ≈ scen.y
            end
            @testset "Cotangent value" begin
                @test dx1 ≈ scen.res1
                @test dx2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:pullback,1,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = if seed isa Batch
        [
            prepare_pullback_batched(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_pullback_batched_same_point(f, ba, x, mycopy_random(seed)),
        ]
    else
        [
            prepare_pullback(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_pullback_same_point(f, ba, x, mycopy_random(seed)),
        ]
    end
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        dx1_in = mysimilar(res1)
        dx2_in = mysimilar(res1)

        if seed isa Batch
            y1, dx1 = value_and_pullback_batched!(f, dx1_in, ba, x, seed, extras_tup...)
            dx2 = pullback_batched!(f, dx2_in, ba, x, seed, extras_tup...)
        else
            y1, dx1 = value_and_pullback!(f, dx1_in, ba, x, seed, extras_tup...)
            dx2 = pullback!(f, dx2_in, ba, x, seed, extras_tup...)
        end

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa PullbackExtras
            end
            @testset "Primal value" begin
                @test y1 ≈ scen.y
            end
            @testset "Cotangent value" begin
                @test dx1_in ≈ scen.res1
                @test dx1 ≈ scen.res1
                @test dx2_in ≈ scen.res1
                @test dx2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:pullback,2,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)
    f! = f

    extras_candidates = if seed isa Batch
        [
            prepare_pullback_batched(
                f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(seed)
            ),
            prepare_pullback_batched_same_point(
                f!, mysimilar(y), ba, x, mycopy_random(seed)
            ),
        ]
    else
        [
            prepare_pullback(f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(seed)),
            prepare_pullback_same_point(f!, mysimilar(y), ba, x, mycopy_random(seed)),
        ]
    end
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1_in = mysimilar(y)
        y2_in = mysimilar(y)

        if seed isa Batch
            y1, dx1 = value_and_pullback_batched(f!, y1_in, ba, x, seed, extras_tup...)
            dx2 = pullback_batched(f!, y2_in, ba, x, seed, extras_tup...)
        else
            y1, dx1 = value_and_pullback(f!, y1_in, ba, x, seed, extras_tup...)
            dx2 = pullback(f!, y2_in, ba, x, seed, extras_tup...)
        end

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa PullbackExtras
            end
            @testset "Primal value" begin
                @test y1_in ≈ scen.y
                @test y1 ≈ scen.y
            end
            @testset "Cotangent value" begin
                @test dx1 ≈ scen.res1
                @test dx2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:pullback,2,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)
    f! = f

    extras_candidates = if seed isa Batch
        [
            prepare_pullback_batched(
                f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(seed)
            ),
            prepare_pullback_batched_same_point(
                f!, mysimilar(y), ba, x, mycopy_random(seed)
            ),
        ]
    else
        [
            prepare_pullback(f!, mysimilar(y), ba, mycopy_random(x), mycopy_random(seed)),
            prepare_pullback_same_point(f!, mysimilar(y), ba, x, mycopy_random(seed)),
        ]
    end
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1_in, dx1_in = mysimilar(y), mysimilar(res1)
        y2_in, dx2_in = mysimilar(y), mysimilar(res1)

        if seed isa Batch
            y1, dx1 = value_and_pullback_batched!(
                f!, y1_in, dx1_in, ba, x, seed, extras_tup...
            )
            dx2 = pullback_batched!(f!, y2_in, dx2_in, ba, x, seed, extras_tup...)
        else
            y1, dx1 = value_and_pullback!(f!, y1_in, dx1_in, ba, x, seed, extras_tup...)
            dx2 = pullback!(f!, y2_in, dx2_in, ba, x, seed, extras_tup...)
        end

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa PullbackExtras
            end
            @testset "Primal value" begin
                @test y1_in ≈ scen.y
                @test y1 ≈ scen.y
            end
            @testset "Cotangent value" begin
                @test dx1_in ≈ scen.res1
                @test dx1 ≈ scen.res1
                @test dx2_in ≈ scen.res1
                @test dx2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

## Derivative

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:derivative,1,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = [prepare_derivative(f, ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1, der1 = value_and_derivative(f, ba, x, extras_tup...)
        der2 = derivative(f, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa DerivativeExtras
            end
            @testset "Primal value" begin
                @test y1 ≈ scen.y
            end
            @testset "Derivative value" begin
                @test der1 ≈ scen.res1
                @test der2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:derivative,1,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = [prepare_derivative(f, ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        der1_in = mysimilar(y)
        y1, der1 = value_and_derivative!(f, der1_in, ba, x, extras_tup...)

        der2_in = mysimilar(y)
        der2 = derivative!(f, der2_in, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa DerivativeExtras
            end
            @testset "Primal value" begin
                @test y1 ≈ scen.y
            end
            @testset "Derivative value" begin
                @test der1_in ≈ scen.res1
                @test der1 ≈ scen.res1
                @test der2_in ≈ scen.res1
                @test der2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:derivative,2,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)
    f! = f

    extras_candidates = [prepare_derivative(f!, mysimilar(y), ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1_in = mysimilar(y)
        y1, der1 = value_and_derivative(f!, y1_in, ba, x, extras_tup...)

        y2_in = mysimilar(y)
        der2 = derivative(f!, y2_in, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa DerivativeExtras
            end
            @testset "Primal value" begin
                @test y1_in ≈ scen.y
                @test y1 ≈ scen.y
            end
            @testset "Derivative value" begin
                @test der1 ≈ scen.res1
                @test der2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:derivative,2,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)
    f! = f

    extras_candidates = [prepare_derivative(f!, mysimilar(y), ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1_in, der1_in = mysimilar(y), mysimilar(y)
        y1, der1 = value_and_derivative!(f!, y1_in, der1_in, ba, x, extras_tup...)

        y2_in, der2_in = mysimilar(y), mysimilar(y)
        der2 = derivative!(f!, y2_in, der2_in, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa DerivativeExtras
            end
            @testset "Primal value" begin
                @test y1_in ≈ scen.y
                @test y1 ≈ scen.y
            end
            @testset "Derivative value" begin
                @test der1_in ≈ scen.res1
                @test der1 ≈ scen.res1
                @test der2_in ≈ scen.res1
                @test der2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

## Gradient

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:gradient,1,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = [prepare_gradient(f, ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1, grad1 = value_and_gradient(f, ba, x, extras_tup...)

        grad2 = gradient(f, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa GradientExtras
            end
            @testset "Primal value" begin
                @test y1 ≈ scen.y
            end
            @testset "Gradient value" begin
                @test grad1 ≈ scen.res1
                @test grad2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:gradient,1,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = [prepare_gradient(f, ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        grad1_in = mysimilar(x)
        y1, grad1 = value_and_gradient!(f, grad1_in, ba, x, extras_tup...)

        grad2_in = mysimilar(x)
        grad2 = gradient!(f, grad2_in, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa GradientExtras
            end
            @testset "Primal value" begin
                @test y1 ≈ scen.y
            end
            @testset "Gradient value" begin
                @test grad1_in ≈ scen.res1
                @test grad1 ≈ scen.res1
                @test grad2_in ≈ scen.res1
                @test grad2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

## Jacobian

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:jacobian,1,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = [prepare_jacobian(f, ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1, jac1 = value_and_jacobian(f, ba, x, extras_tup...)

        jac2 = jacobian(f, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa JacobianExtras
            end
            @testset "Primal value" begin
                @test y1 ≈ scen.y
            end
            @testset "Jacobian value" begin
                @test jac1 ≈ scen.res1
                @test jac2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:jacobian,1,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = [prepare_jacobian(f, ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        jac1_in = mysimilar(new_scen.res1)
        y1, jac1 = value_and_jacobian!(f, jac1_in, ba, x, extras_tup...)

        jac2_in = mysimilar(new_scen.res1)
        jac2 = jacobian!(f, jac2_in, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa JacobianExtras
            end
            @testset "Primal value" begin
                @test y1 ≈ scen.y
            end
            @testset "Jacobian value" begin
                @test jac1_in ≈ scen.res1
                @test jac1 ≈ scen.res1
                @test jac2_in ≈ scen.res1
                @test jac2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:jacobian,2,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)
    f! = f

    extras_candidates = [prepare_jacobian(f!, mysimilar(y), ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1_in = mysimilar(y)
        y1, jac1 = value_and_jacobian(f!, y1_in, ba, x, extras_tup...)

        y2_in = mysimilar(y)
        jac2 = jacobian(f!, y2_in, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa JacobianExtras
            end
            @testset "Primal value" begin
                @test y1_in ≈ scen.y
                @test y1 ≈ scen.y
            end
            @testset "Jacobian value" begin
                @test jac1 ≈ scen.res1
                @test jac2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:jacobian,2,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)
    f! = f

    extras_candidates = [prepare_jacobian(f!, mysimilar(y), ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        y1_in, jac1_in = mysimilar(y), mysimilar(new_scen.res1)
        y1, jac1 = value_and_jacobian!(f!, y1_in, jac1_in, ba, x, extras_tup...)

        y2_in, jac2_in = mysimilar(y), mysimilar(new_scen.res1)
        jac2 = jacobian!(f!, y2_in, jac2_in, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa JacobianExtras
            end
            @testset "Primal value" begin
                @test y1_in ≈ scen.y
                @test y1 ≈ scen.y
            end
            @testset "Jacobian value" begin
                @test jac1_in ≈ scen.res1
                @test jac1 ≈ scen.res1
                @test jac2_in ≈ scen.res1
                @test jac2 ≈ scen.res1
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

## Second derivative

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:second_derivative,1,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = [prepare_second_derivative(f, ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        der21 = second_derivative(f, ba, x, extras_tup...)
        y2, der12, der22 = value_derivative_and_second_derivative(f, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa SecondDerivativeExtras
            end
            @testset "Primal value" begin
                @test y2 ≈ scen.y
            end
            @testset "First derivative value" begin
                @test der12 ≈ scen.res1
            end
            @testset "Second derivative value" begin
                @test der21 ≈ scen.res2
                @test der22 ≈ scen.res2
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:second_derivative,1,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = [prepare_second_derivative(f, ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        der21_in = mysimilar(y)
        der21 = second_derivative!(f, der21_in, ba, x, extras_tup...)

        der12_in, der22_in = mysimilar(y), mysimilar(y)
        y2, der12, der22 = value_derivative_and_second_derivative!(
            f, der12_in, der22_in, ba, x, extras_tup...
        )

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa SecondDerivativeExtras
            end
            @testset "Primal value" begin
                @test y2 ≈ scen.y
            end
            @testset "Derivative value" begin
                @test der12_in ≈ scen.res1
                @test der12 ≈ scen.res1
            end
            @testset "Second derivative value" begin
                @test der21_in ≈ scen.res2
                @test der22_in ≈ scen.res2
                @test der21 ≈ scen.res2
                @test der22 ≈ scen.res2
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

## Hessian-vector product

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:hvp,1,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = if seed isa Batch
        [
            prepare_hvp_batched(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_hvp_batched_same_point(f, ba, x, mycopy_random(seed)),
        ]
    else
        [
            prepare_hvp(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_hvp_same_point(f, ba, x, mycopy_random(seed)),
        ]
    end
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        if seed isa Batch
            dg1 = hvp_batched(f, ba, x, seed, extras_tup...)
        else
            dg1 = hvp(f, ba, x, seed, extras_tup...)
        end

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa HVPExtras
            end
            @testset "HVP value" begin
                @test dg1 ≈ scen.res2
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:hvp,1,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = if seed isa Batch
        [
            prepare_hvp_batched(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_hvp_batched_same_point(f, ba, x, mycopy_random(seed)),
        ]
    else
        [
            prepare_hvp(f, ba, mycopy_random(x), mycopy_random(seed)),
            prepare_hvp_same_point(f, ba, x, mycopy_random(seed)),
        ]
    end
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        dg1_in = mysimilar(res2)

        if seed isa Batch
            dg1 = hvp_batched!(f, dg1_in, ba, x, seed, extras_tup...)
        else
            dg1 = hvp!(f, dg1_in, ba, x, seed, extras_tup...)
        end

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa HVPExtras
            end
            @testset "HVP value" begin
                @test dg1_in ≈ scen.res2
                @test dg1 ≈ scen.res2
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

## Hessian

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:hessian,1,:outofplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = [prepare_hessian(f, ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        hess1 = hessian(f, ba, x, extras_tup...)
        y2, grad2, hess2 = value_gradient_and_hessian(f, ba, x, extras_tup...)

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa HessianExtras
            end
            @testset "Primal value" begin
                @test y2 ≈ scen.y
            end
            @testset "Gradient value" begin
                @test grad2 ≈ scen.res1
            end
            @testset "Hessian value" begin
                @test hess1 ≈ scen.res2
                @test hess2 ≈ scen.res2
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::Scenario{:hessian,1,:inplace};
    isequal::Function,
    isapprox::Function,
    atol,
    rtol,
)
    @compat (; f, x, y, seed, res1, res2) = new_scen = deepcopy(scen)

    extras_candidates = [prepare_hessian(f, ba, mycopy_random(x))]
    extras_tup_candidates = vcat((), tuple.(extras_candidates))

    @testset "$(testset_name(k))" for (k, extras_tup) in enumerate(extras_tup_candidates)
        hess1_in = mysimilar(new_scen.res2)
        hess1 = hessian!(f, hess1_in, ba, x, extras_tup...)
        grad2_in, hess2_in = mysimilar(new_scen.res1), mysimilar(new_scen.res2)
        y2, grad2, hess2 = value_gradient_and_hessian!(
            f, grad2_in, hess2_in, ba, x, extras_tup...
        )

        let (≈)(x, y) = isapprox(x, y; atol, rtol)
            @testset "Extras type" begin
                @test isempty(extras_tup) || only(extras_tup) isa HessianExtras
            end
            @testset "Primal value" begin
                @test y2 ≈ scen.y
            end
            @testset "Gradient value" begin
                @test grad2_in ≈ scen.res1
                @test grad2 ≈ scen.res1
            end
            @testset "Hessian value" begin
                @test hess1_in ≈ scen.res2
                @test hess2_in ≈ scen.res2
                @test hess1 ≈ scen.res2
                @test hess2 ≈ scen.res2
            end
        end
    end
    test_scen_intact(new_scen, scen; isequal)
    return nothing
end
