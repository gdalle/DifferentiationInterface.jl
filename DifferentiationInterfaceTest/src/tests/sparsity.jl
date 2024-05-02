## Jacobian

function test_sparsity(
    ba::AbstractADType, scen::JacobianScenario{1,:outofplace}; ref_backend
)
    @unpack f, x, y = new_scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_true = if ref_backend isa AbstractADType
        jacobian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    _, jac1 = value_and_jacobian(f, ba, x, extras)
    jac2 = jacobian(f, ba, x, extras)

    @testset "Sparsity pattern" begin
        @test nnz(jac1) == nnz(jac_true)
        @test nnz(jac2) == nnz(jac_true)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::JacobianScenario{1,:inplace}; ref_backend)
    @unpack f, x, y = new_scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_true = if ref_backend isa AbstractADType
        jacobian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    _, jac1 = value_and_jacobian!(f, mysimilar(jac_true), ba, x, extras)
    jac2 = jacobian!(f, mysimilar(jac_true), ba, x, extras)

    @testset "Sparsity pattern" begin
        @test nnz(jac1) == nnz(jac_true)
        @test nnz(jac2) == nnz(jac_true)
    end
    return nothing
end

function test_sparsity(
    ba::AbstractADType, scen::JacobianScenario{2,:outofplace}; ref_backend
)
    @unpack f, x, y = new_scen = deepcopy(scen)
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
        @test nnz(jac1) == nnz(jac_true)
        @test nnz(jac2) == nnz(jac_true)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::JacobianScenario{2,:inplace}; ref_backend)
    @unpack f, x, y = new_scen = deepcopy(scen)
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
        @test nnz(jac1) == nnz(jac_true)
        @test nnz(jac2) == nnz(jac_true)
    end
    return nothing
end

## Hessian

function test_sparsity(
    ba::AbstractADType, scen::HessianScenario{1,:outofplace}; ref_backend
)
    @unpack f, x, y = new_scen = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    hess_true = if ref_backend isa AbstractADType
        hessian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    hess1 = hessian(f, ba, x, extras)

    @testset "Sparsity pattern" begin
        @test nnz(hess1) == nnz(hess_true)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::HessianScenario{1,:inplace}; ref_backend)
    @unpack f, x, y = new_scen = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    hess_true = if ref_backend isa AbstractADType
        hessian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    hess1 = hessian!(f, mysimilar(hess_true), ba, x, extras)

    @testset "Sparsity pattern" begin
        @test nnz(hess1) == nnz(hess_true)
    end
    return nothing
end
