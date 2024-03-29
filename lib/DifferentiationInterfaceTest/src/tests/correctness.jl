## No overwrite

function test_scen_intact(new_scen, scen)
    @testset "Scenario intact" begin
        @test new_scen.x == scen.x
        @test new_scen.y == scen.y
        @test new_scen.dx == scen.dx
        @test new_scen.dy == scen.dy
    end
end

## Pushforward

function test_correctness(
    ba::AbstractADType,
    ::typeof(pushforward),
    scen::Scenario{false};
    isapprox::Function,
    atol,
    rtol,
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    dy_true = if ref isa AbstractADType
        pushforward(f, ref, x, dx)
    else
        ref.pushforward(x, dx)
    end

    y1, dy1 = value_and_pushforward(f, ba, x, dx, extras)
    y2, dy2 = value_and_pushforward!!(f, mysimilar(dy), ba, x, dx, extras)

    dy3 = pushforward(f, ba, x, dx, extras)
    dy4 = pushforward!!(f, mysimilar(dy), ba, x, dx, extras)

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
    ::typeof(pushforward),
    scen::Scenario{true};
    isapprox::Function,
    atol,
    rtol,
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    dy_true = if ref isa AbstractADType
        last(value_and_pushforward!!(f!, mysimilar(y), mysimilar(dy), ref, x, dx))
    else
        ref.pushforward(x, dx)
    end

    y10 = mysimilar(y)
    y1, dy1 = value_and_pushforward!!(f!, y10, mysimilar(dy), ba, x, dx, extras)

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
    ::typeof(pullback),
    scen::Scenario{false};
    isapprox::Function,
    atol,
    rtol,
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    dx_true = if ref isa AbstractADType
        pullback(f, ref, x, dy)
    else
        ref.pullback(x, dy)
    end

    y1, dx1 = value_and_pullback(f, ba, x, dy, extras)
    y2, dx2 = value_and_pullback!!(f, mysimilar(dx), ba, x, dy, extras)

    dx3 = pullback(f, ba, x, dy, extras)
    dx4 = pullback!!(f, mysimilar(dx), ba, x, dy, extras)

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
    ::typeof(pullback),
    scen::Scenario{true};
    isapprox::Function,
    atol,
    rtol,
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    dx_true = if ref isa AbstractADType
        last(value_and_pullback!!(f, mysimilar(y), mysimilar(dx), ref, x, dy))
    else
        ref.pullback(x, dy)
    end

    y10 = mysimilar(y)
    y1, dx1 = value_and_pullback!!(f!, y10, mysimilar(dx), ba, x, dy, extras)

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
    ::typeof(derivative),
    scen::Scenario{false};
    isapprox::Function,
    atol,
    rtol,
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    der_true = if ref isa AbstractADType
        derivative(f, ref, x)
    else
        ref.derivative(x)
    end

    y1, der1 = value_and_derivative(f, ba, x, extras)
    y2, der2 = value_and_derivative!!(f, mysimilar(dy), ba, x, extras)

    der3 = derivative(f, ba, x, extras)
    der4 = derivative!!(f, mysimilar(dy), ba, x, extras)

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
    ::typeof(derivative),
    scen::Scenario{true};
    isapprox::Function,
    atol,
    rtol,
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    der_true = if ref isa AbstractADType
        last(value_and_derivative!!(f!, mysimilar(y), mysimilar(dy), ref, x))
    else
        ref.derivative(x)
    end

    y10 = mysimilar(y)
    y1, der1 = value_and_derivative!!(f!, y10, mysimilar(dy), ba, x, extras)

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
    ::typeof(gradient),
    scen::Scenario{false};
    isapprox::Function,
    atol,
    rtol,
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    grad_true = if ref isa AbstractADType
        gradient(f, ref, x)
    else
        ref.gradient(x)
    end

    y1, grad1 = value_and_gradient(f, ba, x, extras)
    y2, grad2 = value_and_gradient!!(f, mysimilar(dx), ba, x, extras)

    grad3 = gradient(f, ba, x, extras)
    grad4 = gradient!!(f, mysimilar(dx), ba, x, extras)

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
    ::typeof(jacobian),
    scen::Scenario{false};
    isapprox::Function,
    atol,
    rtol,
)
    (; f, x, y, ref) = new_scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_true = if ref isa AbstractADType
        jacobian(f, ref, x)
    else
        ref.jacobian(x)
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
    ::typeof(jacobian),
    scen::Scenario{true};
    isapprox::Function,
    atol,
    rtol,
)
    (; f, x, y, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    jac_shape = Matrix{eltype(y)}(undef, length(y), length(x))
    jac_true = if ref isa AbstractADType
        last(value_and_jacobian!!(f!, mysimilar(y), mysimilar(jac_shape), ref, x))
    else
        ref.jacobian(x)
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
    ::typeof(second_derivative),
    scen::Scenario;
    isapprox::Function,
    atol,
    rtol,
)
    (; f, x, dy, ref) = new_scen = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)
    der2_true = if ref isa AbstractADType
        second_derivative(f, ref, x)
    else
        ref.second_derivative(x)
    end

    der21 = second_derivative(f, ba, x, extras)
    der22 = second_derivative!!(f, mysimilar(dy), ba, x, extras)

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
    ba::AbstractADType, ::typeof(hvp), scen::Scenario; isapprox::Function, atol, rtol
)
    (; f, x, dx, ref) = new_scen = deepcopy(scen)
    extras = prepare_hvp(f, ba, x)
    hvp_true = if ref isa AbstractADType
        hvp(f, ref, x, dx)
    else
        ref.hvp(x, dx)
    end

    hvp1 = hvp(f, ba, x, dx, extras)
    hvp2 = hvp!!(f, mysimilar(dx), ba, x, dx, extras)

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
    ba::AbstractADType, ::typeof(hessian), scen::Scenario; isapprox::Function, atol, rtol
)
    (; f, x, y, ref) = new_scen = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    hess_true = if ref isa AbstractADType
        hessian(f, ref, x)
    else
        ref.hessian(x)
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
