## Jacobian

function test_sparsity(ba::AbstractADType, scen::JacobianScenario{false}; ref_backend)
    (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_true = if ref_backend isa AbstractADType
        jacobian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    _, jac1 = value_and_jacobian(f, ba, x, extras)
    _, jac2 = value_and_jacobian!!(f, mysimilar(jac_true), ba, x, extras)

    jac3 = jacobian(f, ba, x, extras)
    jac4 = jacobian!!(f, mysimilar(jac_true), ba, x, extras)

    @testset "Sparse type" begin
        @test jac1 isa SparseMatrixCSC
        @test jac2 isa SparseMatrixCSC
        @test jac3 isa SparseMatrixCSC
        @test jac4 isa SparseMatrixCSC
    end
    @testset "Sparsity pattern" begin
        @test nnz(jac1) < length(jac_true)
        @test nnz(jac2) < length(jac_true)
        @test nnz(jac3) < length(jac_true)
        @test nnz(jac4) < length(jac_true)
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::JacobianScenario{true}; ref_backend)
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
    _, jac1 = value_and_jacobian!!(f!, y10, mysimilar(jac_true), ba, x, extras)

    @testset "Sparse type" begin
        @test jac1 isa SparseMatrixCSC
    end
    @testset "Sparsity pattern" begin
        @test nnz(jac1) < length(jac_true)
    end
    return nothing
end

## Hessian

function test_sparsity(ba::AbstractADType, scen::HessianScenario{false}; ref_backend)
    (; f, x, y) = new_scen = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    hess_true = if ref_backend isa AbstractADType
        hessian(f, ref_backend, x)
    else
        new_scen.ref(x)
    end

    hess1 = hessian(f, ba, x, extras)
    hess2 = hessian!!(f, mysimilar(hess_true), ba, x, extras)

    @testset "Sparse type" begin
        @test hess1 isa SparseMatrixCSC
        @test hess2 isa SparseMatrixCSC
    end
    @testset "Sparsity pattern" begin
        @test nnz(hess1) < length(hess_true)
        @test nnz(hess2) < length(hess_true)
    end
    return nothing
end
