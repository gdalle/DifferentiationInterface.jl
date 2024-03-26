## No overwrite

function test_scen_intact(new_scen, scen)
    let (≈)(x, y) = myisapprox(x, y; rtol=0)
        @testset "Scenario intact" begin
            @test new_scen.x ≈ scen.x
            @test new_scen.y ≈ scen.y
            @test new_scen.dx ≈ scen.dx
            @test new_scen.dy ≈ scen.dy
        end
    end
end

## Pushforward

function test_correctness(
    ba::AbstractADType, ::typeof(pushforward), scen::Scenario{false}; rtol
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    dy_true = if ref isa AbstractADType
        pushforward(f, ref, x, dx)
    else
        ref.pushforward(x, dx)
    end

    y1, dy1 = value_and_pushforward(f, ba, x, dx)
    y2, dy2 = value_and_pushforward!!(f, mysimilar(dy), ba, x, dx)

    dy3 = pushforward(f, ba, x, dx)
    dy4 = pushforward!!(f, mysimilar(dy), ba, x, dx)

    let (≈)(x, y) = myisapprox(x, y; rtol)
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
    ba::AbstractADType, ::typeof(pushforward), scen::Scenario{true}; rtol
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    dy_true = if ref isa AbstractADType
        last(value_and_pushforward!!(f!, mysimilar(y), mysimilar(dy), ref, x, dx))
    else
        ref.pushforward(x, dx)
    end

    y10 = mysimilar(y)
    y1, dy1 = value_and_pushforward!!(f!, y10, mysimilar(dy), ba, x, dx)

    let (≈)(x, y) = myisapprox(x, y; rtol)
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
    ba::AbstractADType, ::typeof(pullback), scen::Scenario{false}; rtol
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    dx_true = if ref isa AbstractADType
        pullback(f, ref, x, dy)
    else
        ref.pullback(x, dy)
    end

    y1, dx1 = value_and_pullback(f, ba, x, dy)
    y2, dx2 = value_and_pullback!!(f, mysimilar(dx), ba, x, dy)

    dx3 = pullback(f, ba, x, dy)
    dx4 = pullback!!(f, mysimilar(dx), ba, x, dy)

    let (≈)(x, y) = myisapprox(x, y; rtol)
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
    ba::AbstractADType, ::typeof(pullback), scen::Scenario{true}; rtol
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    dx_true = if ref isa AbstractADType
        last(value_and_pullback!!(f, mysimilar(y), mysimilar(dx), ref, x, dy))
    else
        ref.pullback(x, dy)
    end

    y10 = mysimilar(y)
    y1, dx1 = value_and_pullback!!(f!, y10, mysimilar(dx), ba, x, dy)

    let (≈)(x, y) = myisapprox(x, y; rtol)
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
    ba::AbstractADType, ::typeof(derivative), scen::Scenario{false}; rtol
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    der_true = if ref isa AbstractADType
        derivative(f, ref, x)
    else
        ref.derivative(x)
    end

    y1, der1 = value_and_derivative(f, ba, x)
    y2, der2 = value_and_derivative!!(f, mysimilar(dy), ba, x)

    der3 = derivative(f, ba, x)
    der4 = derivative!!(f, mysimilar(dy), ba, x)

    let (≈)(x, y) = myisapprox(x, y; rtol)
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
    ba::AbstractADType, ::typeof(derivative), scen::Scenario{true}; rtol
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    der_true = if ref isa AbstractADType
        last(value_and_derivative!!(f!, mysimilar(y), mysimilar(dy), ref, x))
    else
        ref.derivative(x)
    end

    y10 = mysimilar(y)
    y1, der1 = value_and_derivative!!(f!, y10, mysimilar(dy), ba, x)

    let (≈)(x, y) = myisapprox(x, y; rtol)
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
    ba::AbstractADType, ::typeof(gradient), scen::Scenario{false}; rtol
)
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    grad_true = if ref isa AbstractADType
        gradient(f, ref, x)
    else
        ref.gradient(x)
    end

    y1, grad1 = value_and_gradient(f, ba, x)
    y2, grad2 = value_and_gradient!!(f, mysimilar(dx), ba, x)

    grad3 = gradient(f, ba, x)
    grad4 = gradient!!(f, mysimilar(dx), ba, x)

    let (≈)(x, y) = myisapprox(x, y; rtol)
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
    ba::AbstractADType, ::typeof(jacobian), scen::Scenario{false}; rtol
)
    (; f, x, y, ref) = new_scen = deepcopy(scen)
    jac_true = if ref isa AbstractADType
        jacobian(f, ref, x)
    else
        ref.jacobian(x)
    end

    y1, jac1 = value_and_jacobian(f, ba, x)
    y2, jac2 = value_and_jacobian!!(f, mysimilar(jac_true), ba, x)

    jac3 = jacobian(f, ba, x)
    jac4 = jacobian!!(f, mysimilar(jac_true), ba, x)

    let (≈)(x, y) = myisapprox(x, y; rtol)
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
    ba::AbstractADType, ::typeof(jacobian), scen::Scenario{true}; rtol
)
    (; f, x, y, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    jac_shape = Matrix{eltype(y)}(undef, length(y), length(x))
    jac_true = if ref isa AbstractADType
        last(value_and_jacobian!!(f!, mysimilar(y), mysimilar(jac_shape), ref, x))
    else
        ref.jacobian(x)
    end

    y10 = mysimilar(y)
    y1, jac1 = value_and_jacobian!!(f!, y10, mysimilar(jac_true), ba, x)

    let (≈)(x, y) = myisapprox(x, y; rtol)
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
    ba::AbstractADType, ::typeof(second_derivative), scen::Scenario; rtol
)
    (; f, x, ref) = new_scen = deepcopy(scen)
    der2_true = if ref isa AbstractADType
        second_derivative(f, ref, x)
    else
        ref.second_derivative(x)
    end

    der21 = second_derivative(f, ba, x)

    let (≈)(x, y) = myisapprox(x, y; rtol)
        @testset "Second derivative value" begin
            @test der21 ≈ der2_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Hessian-vector product

function test_correctness(ba::AbstractADType, ::typeof(hvp), scen::Scenario; rtol)
    (; f, x, dx, ref) = new_scen = deepcopy(scen)
    hvp_true = if ref isa AbstractADType
        hvp(f, ref, x, dx)
    else
        ref.hvp(x, dx)
    end

    hvp1 = hvp(f, ba, x, dx)

    let (≈)(x, y) = myisapprox(x, y; rtol)
        @testset "HVP value" begin
            @test hvp1 ≈ hvp_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Hessian

function test_correctness(ba::AbstractADType, ::typeof(hessian), scen::Scenario; rtol)
    (; f, x, y, ref) = new_scen = deepcopy(scen)
    hess_true = if ref isa AbstractADType
        hessian(f, ref, x)
    else
        ref.hessian(x)
    end

    hess1 = hessian(f, ba, x)

    let (≈)(x, y) = myisapprox(x, y; rtol)
        @testset "Hessian value" begin
            @test hess1 ≈ hess_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end
