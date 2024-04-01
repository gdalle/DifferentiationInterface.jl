function test_sparsity(ba::AbstractADType, ::typeof(jacobian), scen::Scenario{false};)
    (; f, x, y, ref) = new_scen = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_true = if ref isa AbstractADType
        jacobian(f, ref, x)
    else
        ref.jacobian(x)
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

function test_sparsity(ba::AbstractADType, ::typeof(jacobian), scen::Scenario{true};)
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
    _, jac1 = value_and_jacobian!!(f!, y10, mysimilar(jac_true), ba, x, extras)

    @testset "Sparse type" begin
        @test jac1 isa SparseMatrixCSC
    end
    @testset "Sparsity pattern" begin
        @test nnz(jac1) < length(jac_true)
    end
    return nothing
end
