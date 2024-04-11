## No overwrite

function test_scen_intact(new_scen, scen)
    @testset "Scenario intact" begin
        @test new_scen.x == scen.x
        @test new_scen.y == scen.y
        if hasfield(typeof(scen), :dx)
            @test new_scen.dx == scen.dx
        end
        if hasfield(typeof(scen), :dy)
            @test new_scen.dy == scen.dy
        end
    end
end

## Pushforward

function test_correctness(
    ba::AbstractADType,
    scen::PushforwardScenario{1,:outofplace};
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
    dy2 = pushforward(f, ba, x, dx, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Tangent value" begin
            @test dy1 ≈ dy_true
            @test dy2 ≈ dy_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PushforwardScenario{1,:inplace};
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

    dy1_in = mysimilar(y)
    y1, dy1 = value_and_pushforward!(f, dy1_in, ba, x, dx, extras)

    dy2_in = mysimilar(y)
    dy2 = pushforward!(f, dy2_in, ba, x, dx, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Tangent value" begin
            @test dy1_in ≈ dy_true
            @test dy1 ≈ dy_true
            @test dy2_in ≈ dy_true
            @test dy2 ≈ dy_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PushforwardScenario{2,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y, dx) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    dy_true = if ref_backend isa AbstractADType
        pushforward!(f!, (mysimilar(y), mysimilar(y)), ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    y1_in, dy1_in = mysimilar(y), mysimilar(y)
    y1, dy1 = value_and_pushforward!(f!, (y1_in, dy1_in), ba, x, dx, extras)

    y2_in, dy2_in = mysimilar(y), mysimilar(y)
    dy2 = pushforward!(f!, (y2_in, dy2_in), ba, x, dx, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1_in ≈ y
            @test y1 ≈ y
        end
        @testset "Tangent value" begin
            @test dy1_in ≈ dy_true
            @test dy1 ≈ dy_true
            @test dy2_in ≈ dy_true
            @test dy2 ≈ dy_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Pullback

function test_correctness(
    ba::AbstractADType,
    scen::PullbackScenario{1,:outofplace};
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

    dx2 = pullback(f, ba, x, dy, extras)

    y3, pullbackfunc = value_and_pullback_split(f, ba, x, extras)
    pullbackfunc(dy)  # call once in case the second errors
    dx3 = pullbackfunc(dy)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
            @test y3 ≈ y
        end
        @testset "Cotangent value" begin
            @test dx1 ≈ dx_true
            @test dx2 ≈ dx_true
            @test dx3 ≈ dx_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PullbackScenario{1,:inplace};
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

    dx1_in = mysimilar(x)
    y1, dx1 = value_and_pullback!(f, dx1_in, ba, x, dy, extras)

    dx2_in = mysimilar(x)
    dx2 = pullback!(f, dx2_in, ba, x, dy, extras)

    y3, pullbackfunc! = value_and_pullback!_split(f, ba, x, extras)
    pullbackfunc!(mysimilar(x), dy)  # call once in case the second errors
    dx3_in = mysimilar(x)
    dx3 = pullbackfunc!(dx3_in, dy)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
            @test y3 ≈ y
        end
        @testset "Cotangent value" begin
            @test dx1_in ≈ dx_true
            @test dx1 ≈ dx_true
            @test dx2_in ≈ dx_true
            @test dx2 ≈ dx_true
            @test dx3_in ≈ dx_true
            @test dx3 ≈ dx_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::PullbackScenario{2,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y, dy) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    dx_true = if ref_backend isa AbstractADType
        pullback!(f, (mysimilar(y), mysimilar(x)), ref_backend, x, dy)
    else
        new_scen.ref(x, dy)
    end

    y1_in, dx1_in = mysimilar(y), mysimilar(x)
    y1, dx1 = value_and_pullback!(f!, (y1_in, dx1_in), ba, x, dy, extras)

    y2_in, dx2_in = mysimilar(y), mysimilar(x)
    dx2 = pullback!(f!, (y2_in, dx2_in), ba, x, dy, extras)

    y3_in = mysimilar(y)
    y3, pullbackfunc! = value_and_pullback!_split!(f!, y3_in, ba, x, extras)
    pullbackfunc!((mysimilar(y), mysimilar(x)), dy)  # call once in case the second errors
    y3_in2, dx3_in = mysimilar(y), mysimilar(x)
    dx3 = pullbackfunc!((y3_in2, dx3_in), dy)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1_in ≈ y
            @test y1 ≈ y
            @test y3_in ≈ y
            @test y3 ≈ y
        end
        @testset "Cotangent value" begin
            @test dx1 ≈ dx_true
            @test dx2 ≈ dx_true
            @test dx3 ≈ dx_true
            if !isa(dx_true, Number)  # TODO: find cleaner fix
                @test dx1_in ≈ dx_true
                @test dx2_in ≈ dx_true
                @test dx3_in ≈ dx_true
            end
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Derivative

function test_correctness(
    ba::AbstractADType,
    scen::DerivativeScenario{1,:outofplace};
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

    der2 = derivative(f, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Derivative value" begin
            @test der1 ≈ der_true
            @test der2 ≈ der_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::DerivativeScenario{1,:inplace};
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

    der1_in = mysimilar(y)
    y1, der1 = value_and_derivative!(f, der1_in, ba, x, extras)

    der2_in = mysimilar(y)
    der2 = derivative!(f, der2_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Derivative value" begin
            @test der1_in ≈ der_true
            @test der1 ≈ der_true
            @test der2_in ≈ der_true
            @test der2 ≈ der_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::DerivativeScenario{2,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    der_true = if ref_backend isa AbstractADType
        derivative!(f!, (mysimilar(y), mysimilar(y)), ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1_in, der1_in = mysimilar(y), mysimilar(y)
    y1, der1 = value_and_derivative!(f!, (y1_in, der1_in), ba, x, extras)

    y2_in, der2_in = mysimilar(y), mysimilar(y)
    der2 = derivative!(f!, (y2_in, der2_in), ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1_in ≈ y
            @test y1 ≈ y
        end
        @testset "Derivative value" begin
            @test der1_in ≈ der_true
            @test der1 ≈ der_true
            @test der2_in ≈ der_true
            @test der2 ≈ der_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Gradient

function test_correctness(
    ba::AbstractADType,
    scen::GradientScenario{1,:outofplace};
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

    grad2 = gradient(f, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Gradient value" begin
            @test grad1 ≈ grad_true
            @test grad2 ≈ grad_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::GradientScenario{1,:inplace};
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

    grad1_in = mysimilar(x)
    y1, grad1 = value_and_gradient!(f, grad1_in, ba, x, extras)

    grad2_in = mysimilar(x)
    grad2 = gradient!(f, grad2_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Gradient value" begin
            @test grad1_in ≈ grad_true
            @test grad1 ≈ grad_true
            @test grad2_in ≈ grad_true
            @test grad2 ≈ grad_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Jacobian

function test_correctness(
    ba::AbstractADType,
    scen::JacobianScenario{1,:outofplace};
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

    jac2 = jacobian(f, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Jacobian value" begin
            @test jac1 ≈ jac_true
            @test jac2 ≈ jac_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::JacobianScenario{1,:inplace};
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

    jac1_in = mysimilar(jac_true)
    y1, jac1 = value_and_jacobian!(f, jac1_in, ba, x, extras)

    jac2_in = mysimilar(jac_true)
    jac2 = jacobian!(f, jac2_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1 ≈ y
        end
        @testset "Jacobian value" begin
            @test jac1_in ≈ jac_true
            @test jac1 ≈ jac_true
            @test jac2_in ≈ jac_true
            @test jac2 ≈ jac_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::JacobianScenario{2,:inplace};
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
        jacobian!(f!, (mysimilar(y), mysimilar(jac_shape)), ref_backend, x)
    else
        new_scen.ref(x)
    end

    y1_in, jac1_in = mysimilar(y), mysimilar(jac_true)
    y1, jac1 = value_and_jacobian!(f!, (y1_in, jac1_in), ba, x, extras)

    y2_in, jac2_in = mysimilar(y), mysimilar(jac_true)
    jac2 = jacobian!(f!, (y2_in, jac2_in), ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Primal value" begin
            @test y1_in ≈ y
            @test y1 ≈ y
        end
        @testset "Jacobian value" begin
            @test jac1_in ≈ jac_true
            @test jac1 ≈ jac_true
            @test jac2_in ≈ jac_true
            @test jac2 ≈ jac_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Second derivative

function test_correctness(
    ba::AbstractADType,
    scen::SecondDerivativeScenario{1,:outofplace};
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

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Second derivative value" begin
            @test der21 ≈ der2_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::SecondDerivativeScenario{1,:inplace};
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

    der21_in = mysimilar(y)
    der21 = second_derivative!(f, der21_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Second derivative value" begin
            @test der21_in ≈ der2_true
            @test der21 ≈ der2_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Hessian-vector product

function test_correctness(
    ba::AbstractADType,
    scen::HVPScenario{1,:outofplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, dx) = new_scen = deepcopy(scen)
    extras = prepare_hvp(f, ba, x, dx)
    p_true = if ref_backend isa AbstractADType
        hvp(f, ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    p1 = hvp(f, ba, x, dx, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "HVP value" begin
            @test p1 ≈ p_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::HVPScenario{1,:inplace};
    isapprox::Function,
    atol,
    rtol,
    ref_backend,
)
    (; f, x, dx) = new_scen = deepcopy(scen)
    extras = prepare_hvp(f, ba, x, dx)
    p_true = if ref_backend isa AbstractADType
        hvp(f, ref_backend, x, dx)
    else
        new_scen.ref(x, dx)
    end

    p1_in = mysimilar(x)
    p1 = hvp!(f, p1_in, ba, x, dx, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "HVP value" begin
            @test p1_in ≈ p_true
            @test p1 ≈ p_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

## Hessian

function test_correctness(
    ba::AbstractADType,
    scen::HessianScenario{1,:outofplace};
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

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Hessian value" begin
            @test hess1 ≈ hess_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end

function test_correctness(
    ba::AbstractADType,
    scen::HessianScenario{1,:inplace};
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

    hess1_in = mysimilar(hess_true)
    hess1 = hessian!(f, hess1_in, ba, x, extras)

    let (≈)(x, y) = isapprox(x, y; atol, rtol)
        @testset "Hessian value" begin
            @test hess1_in ≈ hess_true
            @test hess1 ≈ hess_true
        end
    end
    test_scen_intact(new_scen, scen)
    return nothing
end
