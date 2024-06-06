mynnz(A::AbstractMatrix) = nnz(A)
mynnz(A::Union{Transpose,Adjoint}) = nnz(parent(A))  # fix for Julia 1.6

## Jacobian

function test_sparsity(
    ba::AbstractADType, scen::JacobianScenario{1,:outofplace}; ref_backend
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_true = if ref_backend isa AbstractADType
        jacobian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    _, jac1 = value_and_jacobian(f, ba, x, extras)
    jac2 = jacobian(f, ba, x, extras)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(jac_true)
        @test mynnz(jac2) == mynnz(jac_true)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::JacobianScenario{1,:inplace}; ref_backend)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_true = if ref_backend isa AbstractADType
        jacobian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    _, jac1 = value_and_jacobian!(f, mysimilar(jac_true), ba, x, extras)
    jac2 = jacobian!(f, mysimilar(jac_true), ba, x, extras)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(jac_true)
        @test mynnz(jac2) == mynnz(jac_true)
    end
    return nothing
end

function test_sparsity(
    ba::AbstractADType, scen::JacobianScenario{2,:outofplace}; ref_backend
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, mysimilar(y), ba, x)
    jac_true = if ref_backend isa AbstractADType
        jacobian(f!, mysimilar(y), ref_backend, x)
    else
        new_scen.ref(x)
    end

    _, jac1 = value_and_jacobian(f!, mysimilar(y), ba, x, extras)
    jac2 = jacobian(f!, mysimilar(y), ba, x, extras)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(jac_true)
        @test mynnz(jac2) == mynnz(jac_true)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::JacobianScenario{2,:inplace}; ref_backend)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, mysimilar(y), ba, x)
    jac_true = if ref_backend isa AbstractADType
        jacobian(f!, mysimilar(y), ref_backend, x)
    else
        new_scen.ref(x)
    end

    _, jac1 = value_and_jacobian!(f!, mysimilar(y), mysimilar(jac_true), ba, x, extras)
    jac2 = jacobian!(f!, mysimilar(y), mysimilar(jac_true), ba, x, extras)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(jac_true)
        @test mynnz(jac2) == mynnz(jac_true)
    end
    return nothing
end

## Hessian

function test_sparsity(
    ba::AbstractADType, scen::HessianScenario{1,:outofplace}; ref_backend
)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    hess_true = if ref_backend isa AbstractADType
        hessian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    hess1 = hessian(f, ba, x, extras)
    _, _, hess2 = value_gradient_and_hessian(f, ba, x, extras)

    @testset "Sparsity pattern" begin
        @test mynnz(hess1) == mynnz(hess_true)
        @test mynnz(hess2) == mynnz(hess_true)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::HessianScenario{1,:inplace}; ref_backend)
    @compat (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    hess_true = if ref_backend isa AbstractADType
        hessian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    hess1 = hessian!(f, mysimilar(hess_true), ba, x, extras)
    _, _, hess2 = value_gradient_and_hessian!(
        f, mysimilar(x), mysimilar(hess_true), ba, x, extras
    )

    @testset "Sparsity pattern" begin
        @test mynnz(hess1) == mynnz(hess_true)
        @test mynnz(hess2) == mynnz(hess_true)
    end
    return nothing
end
