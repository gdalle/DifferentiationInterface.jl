## No overwrite

function test_scen_intact(new_scen, scen)
    @testset "Scenario intact" begin
        @test new_scen.x == scen.x
        @test new_scen.y == scen.y
        if scen isa PushforwardScenario
            @test new_scen.dx == scen.dx
        elseif scen isa PullbackScenario
            @test new_scen.dy == scen.dy
        end
    end
end

## Pushforward

function test_correctness(
    ba::AbstractADType,
    scen::PushforwardScenario{false};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y, dx) = new_scen = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    dy_true = if ref_backend isa AbstractADType
        pushforward(f, ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    y1, dy1 = value_and_pushforward(f, ba, x, dx, extras)
    y2, dy2 = value_and_pushforward!!(f, mysimilar(y), ba, x, dx, extras)

    dy3 = pushforward(f, ba, x, dx, extras)
    dy4 = pushforward!!(f, mysimilar(y), ba, x, dx, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
            @test y2 ≈ y
        end
        @testset "Tangent value" begin
            @test dy1 ≈ dy_true
            @test dy2 ≈ dy_true
            @test dy3 ≈ dy_true
            @test dy4 ≈ dy_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PushforwardScenario{true};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y, dx) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    dy_true = if ref_backend isa AbstractADType
        last(value_and_pushforward!!(f!, mysimilar(y), mysimilar(y), ref_backend, x, dx))
    else
        new_scen.ref(x, dx)
    end

    y10 = mysimilar(y)
    y1, dy1 = value_and_pushforward!!(f!, y10, mysimilar(y), ba, x, dx, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y10 ≈ y
            @test y1 ≈ y
        end
        @testset "Tangent value" begin
            @test dy1 ≈ dy_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Pullback

function test_correctness(
    ba::AbstractADType,
    scen::PullbackScenario{false};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y, dy) = new_scen = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    dx_true = if ref_backend isa AbstractADType
        pullback(f, ref_backend, x, dy)
    else
        new_scen.ref(x, dy)
    end

    y1, dx1 = value_and_pullback(f, ba, x, dy, extras)
    y2, dx2 = value_and_pullback!!(f, mysimilar(x), ba, x, dy, extras)

    dx3 = pullback(f, ba, x, dy, extras)
    dx4 = pullback!!(f, mysimilar(x), ba, x, dy, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
            @test y2 ≈ y
        end
        @testset "Cotangent value" begin
            @test dx1 ≈ dx_true
            @test dx2 ≈ dx_true
            @test dx3 ≈ dx_true
            @test dx4 ≈ dx_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PullbackScenario{true};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y, dy) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    dx_true = if ref_backend isa AbstractADType
        last(value_and_pullback!!(f, mysimilar(y), mysimilar(x), ref_backend, x, dy))
    else
        new_scen.ref(x, dy)
    end

    y10 = mysimilar(y)
    y1, dx1 = value_and_pullback!!(f!, y10, mysimilar(x), ba, x, dy, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y10 ≈ y
            @test y1 ≈ y
        end
        @testset "Cotangent value" begin
            @test dx1 ≈ dx_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Derivative

function test_correctness(
    ba::AbstractADType,
    scen::DerivativeScenario{false};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    der_true = if ref_backend isa AbstractADType
        derivative(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1, der1 = value_and_derivative(f, ba, x, extras)
    y2, der2 = value_and_derivative!!(f, mysimilar(y), ba, x, extras)

    der3 = derivative(f, ba, x, extras)
    der4 = derivative!!(f, mysimilar(y), ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
            @test y2 ≈ y
        end
        @testset "Derivative value" begin
            @test der1 ≈ der_true
            @test der2 ≈ der_true
            @test der3 ≈ der_true
            @test der4 ≈ der_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::DerivativeScenario{true};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    der_true = if ref_backend isa AbstractADType
        last(value_and_derivative!!(f!, mysimilar(y), mysimilar(y), ref_backend, x))
    else
        new_scen.ref(x)
    end

    y10 = mysimilar(y)
    y1, der1 = value_and_derivative!!(f!, y10, mysimilar(y), ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y10 ≈ y
            @test y1 ≈ y
        end
        @testset "Derivative value" begin
            @test der1 ≈ der_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Gradient

function test_correctness(
    ba::AbstractADType,
    scen::GradientScenario{false};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    grad_true = if ref_backend isa AbstractADType
        gradient(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1, grad1 = value_and_gradient(f, ba, x, extras)
    y2, grad2 = value_and_gradient!!(f, mysimilar(x), ba, x, extras)

    grad3 = gradient(f, ba, x, extras)
    grad4 = gradient!!(f, mysimilar(x), ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
            @test y2 ≈ y
        end
        @testset "Gradient value" begin
            @test grad1 ≈ grad_true
            @test grad2 ≈ grad_true
            @test grad3 ≈ grad_true
            @test grad4 ≈ grad_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Jacobian

function test_correctness(
    ba::AbstractADType,
    scen::JacobianScenario{false};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_true = if ref_backend isa AbstractADType
        jacobian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1, jac1 = value_and_jacobian(f, ba, x, extras)
    y2, jac2 = value_and_jacobian!!(f, mysimilar(jac_true), ba, x, extras)

    jac3 = jacobian(f, ba, x, extras)
    jac4 = jacobian!!(f, mysimilar(jac_true), ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
            @test y2 ≈ y
        end
        @testset "Jacobian value" begin
            @test jac1 ≈ jac_true
            @test jac2 ≈ jac_true
            @test jac3 ≈ jac_true
            @test jac4 ≈ jac_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::JacobianScenario{true};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    jac_shape = Matrix{eltype(y)}(undef, length(y), length(x))
    jac_true = if ref_backend isa AbstractADType
        last(value_and_jacobian!!(f!, mysimilar(y), mysimilar(jac_shape), ref_backend, x))
    else
        new_scen.ref(x)
    end

    y10 = mysimilar(y)
    y1, jac1 = value_and_jacobian!!(f!, y10, mysimilar(jac_true), ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y10 ≈ y
            @test y1 ≈ y
        end
        @testset "Jacobian value" begin
            @test jac1 ≈ jac_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Second derivative

function test_correctness(
    ba::AbstractADType,
    scen::SecondDerivativeScenario{false};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)
    der2_true = if ref_backend isa AbstractADType
        second_derivative(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    der21 = second_derivative(f, ba, x, extras)
    der22 = second_derivative!!(f, mysimilar(y), ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Second derivative value" begin
            @test der21 ≈ der2_true
            @test der22 ≈ der2_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Hessian-vector product

function test_correctness(
    ba::AbstractADType,
    scen::HVPScenario{false};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, dx) = new_scen = deepcopy(scen)
    extras = prepare_hvp(f, ba, x)
    hvp_true = if ref_backend isa AbstractADType
        hvp(f, ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    hvp1 = hvp(f, ba, x, dx, extras)
    hvp2 = hvp!!(f, mysimilar(x), ba, x, dx, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "HVP value" begin
            @test hvp1 ≈ hvp_true
            @test hvp2 ≈ hvp_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Hessian

function test_correctness(
    ba::AbstractADType,
    scen::HessianScenario{false};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    hess_true = if ref_backend isa AbstractADType
        hessian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    hess1 = hessian(f, ba, x, extras)
    hess2 = hessian!!(f, mysimilar(hess_true), ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Hessian value" begin
            @test hess1 ≈ hess_true
            @test hess2 ≈ hess_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end
