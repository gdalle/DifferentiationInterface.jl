mynnz(A::AbstractMatrix) = nnz(A)
mynnz(A::Union{Transpose,Adjoint}) = nnz(parent(A))  # fix for Julia 1.6

## Jacobian

function test_sparsity(ba::AbstractADType, scen::Scenario{:jacobian,1,:outofplace})
    @compat (; f, x, y) = scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)

    _, jac1 = value_and_jacobian(f, extras, ba, x)
    jac2 = jacobian(f, extras, ba, x)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(scen.res1)
        @test mynnz(jac2) == mynnz(scen.res1)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::Scenario{:jacobian,1,:inplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)

    _, jac1 = value_and_jacobian!(f, mysimilar(scen.res1), extras, ba, x)
    jac2 = jacobian!(f, mysimilar(scen.res1), extras, ba, x)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(scen.res1)
        @test mynnz(jac2) == mynnz(scen.res1)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::Scenario{:jacobian,2,:outofplace})
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, mysimilar(y), ba, x)

    _, jac1 = value_and_jacobian(f!, mysimilar(y), extras, ba, x)
    jac2 = jacobian(f!, mysimilar(y), extras, ba, x)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(scen.res1)
        @test mynnz(jac2) == mynnz(scen.res1)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::Scenario{:jacobian,2,:inplace})
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, mysimilar(y), ba, x)

    _, jac1 = value_and_jacobian!(f!, mysimilar(y), mysimilar(scen.res1), extras, ba, x)
    jac2 = jacobian!(f!, mysimilar(y), mysimilar(scen.res1), extras, ba, x)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(scen.res1)
        @test mynnz(jac2) == mynnz(scen.res1)
    end
    return nothing
end

## Hessian

function test_sparsity(ba::AbstractADType, scen::Scenario{:hessian,1,:outofplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)

    hess1 = hessian(f, extras, ba, x)
    _, _, hess2 = value_gradient_and_hessian(f, extras, ba, x)

    @testset "Sparsity pattern" begin
        @test mynnz(hess1) == mynnz(scen.res2)
        @test mynnz(hess2) == mynnz(scen.res2)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::Scenario{:hessian,1,:inplace})
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)

    hess1 = hessian!(f, mysimilar(scen.res2), extras, ba, x)
    _, _, hess2 = value_gradient_and_hessian!(
        f, mysimilar(x), mysimilar(scen.res2), extras, ba, x
    )

    @testset "Sparsity pattern" begin
        @test mynnz(hess1) == mynnz(scen.res2)
        @test mynnz(hess2) == mynnz(scen.res2)
    end
    return nothing
end
