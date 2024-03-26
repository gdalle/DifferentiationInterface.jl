## No overwrite

function test_scen_intact(new_scen, scen)
    @testset "Scenario intact" begin
        @test myisapprox(new_scen.x, scen.x)
        @test myisapprox(new_scen.y, scen.y)
        @test myisapprox(new_scen.dx, scen.dx)
        @test myisapprox(new_scen.dy, scen.dy)
    end
end

## Pushforward

function test_correctness(ba::AbstractADType, ::typeof(pushforward), scen::Scenario{false})
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    dy_true = if ref isa AbstractADType
        pushforward(f, ref, x, dx)
    else
        ref(:pushforward)(x, dx)
    end

    y1, dy1 = value_and_pushforward(f, ba, x, dx)
    y2, dy2 = value_and_pushforward!!(f, mysimilar(dy), ba, x, dx)

    dy3 = pushforward(f, ba, x, dx)
    dy4 = pushforward!!(f, mysimilar(dy), ba, x, dx)

    @testset "Primal value" begin
        @test myisapprox(y1, y)
        @test myisapprox(y2, y)
    end
    @testset "Tangent value" begin
        @test myisapprox(dy1, dy_true; rtol=1e-3)
        @test myisapprox(dy2, dy_true; rtol=1e-3)
        @test myisapprox(dy3, dy_true; rtol=1e-3)
        @test myisapprox(dy4, dy_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness(ba::AbstractADType, ::typeof(pushforward), scen::Scenario{true})
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    dy_true = if ref isa AbstractADType
        last(value_and_pushforward!!(f!, mysimilar(y), mysimilar(dy)), ref, x, dx)
    else
        ref(:pushforward)(x, dx)
    end

    y10 = mysimilar(y)
    y1, dy1 = value_and_pushforward!!(f!, y10, mysimilar(dy), ba, x, dx)

    @testset "Primal value" begin
        @test myisapprox(y10, y)
        @test myisapprox(y1, y)
    end
    @testset "Tangent value" begin
        @test myisapprox(dy1, dy_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

## Pullback

function test_correctness(ba::AbstractADType, ::typeof(pullback), scen::Scenario{false})
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    dx_true = if ref isa AbstractADType
        pullback(f, ref, x, dy)
    else
        ref(:pullback)(x, dy)
    end

    y1, dx1 = value_and_pullback(f, ba, x, dy)
    y2, dx2 = value_and_pullback!!(f, mysimilar(dx), ba, x, dy)

    dx3 = pullback(f, ba, x, dy)
    dx4 = pullback!!(f, mysimilar(dx), ba, x, dy)

    @testset "Primal value" begin
        @test myisapprox(y1, y)
        @test myisapprox(y2, y)
    end
    @testset "Cotangent value" begin
        @test myisapprox(dx1, dx_true; rtol=1e-3)
        @test myisapprox(dx2, dx_true; rtol=1e-3)
        @test myisapprox(dx3, dx_true; rtol=1e-3)
        @test myisapprox(dx4, dx_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness(ba::AbstractADType, ::typeof(pullback), scen::Scenario{true})
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    dx_true = if ref isa AbstractADType
        last(value_and_pullback!!(f, mysimilar(y), mysimilar(dx)), ref, x, dy)
    else
        ref(:pullback)(x, dy)
    end

    y10 = mysimilar(y)
    y1, dx1 = value_and_pullback!!(f!, y10, mysimilar(dx), ba, x, dy)

    @testset "Primal value" begin
        @test myisapprox(y10, y)
        @test myisapprox(y1, y)
    end
    @testset "Cotangent value" begin
        @test myisapprox(dx1, dx_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

## Derivative

function test_correctness(ba::AbstractADType, ::typeof(derivative), scen::Scenario{false})
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    der_true = if ref isa AbstractADType
        derivative(f, ref, x)
    else
        ref(:derivative)(x)
    end

    y1, der1 = value_and_derivative(f, ba, x)
    y2, der2 = value_and_derivative!!(f, mysimilar(dy), ba, x)

    der3 = derivative(f, ba, x)
    der4 = derivative!!(f, mysimilar(dy), ba, x)

    @testset "Primal value" begin
        @test myisapprox(y1, y)
        @test myisapprox(y2, y)
    end
    @testset "Derivative value" begin
        @test myisapprox(der1, der_true; rtol=1e-3)
        @test myisapprox(der2, der_true; rtol=1e-3)
        @test myisapprox(der3, der_true; rtol=1e-3)
        @test myisapprox(der4, der_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness(ba::AbstractADType, ::typeof(derivative), scen::Scenario{true})
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    f! = f
    der_true = if ref isa AbstractADType
        last(value_and_derivative!!(f!, mysimilar(y), mysimilar(dy), ref, x))
    else
        ref(:derivative)(x)
    end

    y10 = mysimilar(y)
    y1, der1 = value_and_derivative!!(f!, y10, mysimilar(dy), ba, x)

    @testset "Primal value" begin
        @test myisapprox(y10, y)
        @test myisapprox(y1, y)
    end
    @testset "Derivative value" begin
        @test myisapprox(der1, der_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

## Gradient

function test_correctness(ba::AbstractADType, ::typeof(gradient), scen::Scenario{false})
    (; f, x, y, dx, dy, ref) = new_scen = deepcopy(scen)
    grad_true = if ref isa AbstractADType
        gradient(f, ref, x)
    else
        ref(:gradient)(x)
    end

    y1, grad1 = value_and_gradient(f, ba, x)
    y2, grad2 = value_and_gradient!!(f, mysimilar(dx), ba, x)

    grad3 = gradient(f, ba, x)
    grad4 = gradient!!(f, mysimilar(dx), ba, x)

    @testset "Primal value" begin
        @test myisapprox(y1, y)
        @test myisapprox(y2, y)
    end
    @testset "Gradient value" begin
        @test myisapprox(grad1, grad_true; rtol=1e-3)
        @test myisapprox(grad2, grad_true; rtol=1e-3)
        @test myisapprox(grad3, grad_true; rtol=1e-3)
        @test myisapprox(grad4, grad_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

## Jacobian

function test_correctness(ba::AbstractADType, ::typeof(jacobian), scen::Scenario{false})
    (; f, x, y, ref) = new_scen = deepcopy(scen)
    jac_true = if ref isa AbstractADType
        jacobian(f, ref, x)
    else
        ref(:jacobian)(x)
    end

    y1, jac1 = value_and_jacobian(f, ba, x)
    y2, jac2 = value_and_jacobian!!(f, mysimilar(jac_true), ba, x)

    jac3 = jacobian(f, ba, x)
    jac4 = jacobian!!(f, mysimilar(jac_true), ba, x)

    @testset "Primal value" begin
        @test myisapprox(y1, y)
        @test myisapprox(y2, y)
    end
    @testset "Jacobian value" begin
        @test myisapprox(jac1, jac_true; rtol=1e-3)
        @test myisapprox(jac2, jac_true; rtol=1e-3)
        @test myisapprox(jac3, jac_true; rtol=1e-3)
        @test myisapprox(jac4, jac_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness(ba::AbstractADType, ::typeof(jacobian), scen::Scenario{true})
    (; f, x, y, ref) = new_scen = deepcopy(scen)
    f! = f
    jac_true = if ref isa AbstractADType
        last(value_and_jacobian!!(f!, mysimilar(y), mysimilar(dy), ref, x))
    else
        ref(:jacobian)(x)
    end

    y10 = mysimilar(y)
    y1, jac1 = value_and_jacobian!!(f!, y10, mysimilar(jac_true), ba, x)

    @testset "Primal value" begin
        @test myisapprox(y10, y)
        @test myisapprox(y1, y)
    end
    @testset "Jacobian value" begin
        @test myisapprox(jac1, jac_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

## Second derivative

function test_correctness(ba::AbstractADType, ::typeof(second_derivative), scen::Scenario)
    (; f, x, ref) = deepcopy(scen)
    der2_true = if ref isa AbstractADType
        second_derivative(f, ref, x)
    else
        ref(:second_derivative)(x)
    end

    der21 = second_derivative(f, ba, x)

    @testset "Second derivative value" begin
        @test myisapprox(der21, der2_true; rtol=1e-3)
    end
end

## Hessian-vector product

function test_correctness(ba::AbstractADType, ::typeof(hvp), scen::Scenario)
    (; f, x, dx, ref) = deepcopy(scen)
    hvp_true = if ref isa AbstractADType
        hvp(f, ref, x, dx)
    else
        ref(:hvp)(x, dx)
    end

    hvp1 = hvp(f, ba, x, dx)

    @testset "Hessian-vector product value" begin
        @test myisapprox(hvp1, hvp_true; rtol=1e-3)
    end
end

## Hessian

function test_correctness(ba::AbstractADType, ::typeof(hessian), scen::Scenario)
    (; f, x, y, ref) = deepcopy(scen)
    hess_true = if ref isa AbstractADType
        hessian(f, ref, x)
    else
        ref(:hessian)(x)
    end

    hess1 = hessian(f, ba, x)

    @testset "Hessian value" begin
        @test myisapprox(hess1, hess_true; rtol=1e-3)
    end
end
