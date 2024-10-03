## Jacobian

function test_sparsity(ba::AbstractADType, scen::Scenario{:jacobian,:out,:out})
    @compat (; f, x, y, contexts) = scen = deepcopy(scen)
    prep = prepare_jacobian(f, ba, x, contexts...)

    _, jac1 = value_and_jacobian(f, prep, ba, x, contexts...)
    jac2 = jacobian(f, prep, ba, x, contexts...)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(scen.res1)
        @test mynnz(jac2) == mynnz(scen.res1)
        @test nameof(typeof(jac1)) == nameof(typeof(SMC.sparsity_pattern(prep)))
        @test nameof(typeof(jac2)) == nameof(typeof(SMC.sparsity_pattern(prep)))
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::Scenario{:jacobian,:in,:out})
    @compat (; f, x, y, contexts) = deepcopy(scen)
    prep = prepare_jacobian(f, ba, x, contexts...)

    _, jac1 = value_and_jacobian!(f, mysimilar(scen.res1), prep, ba, x, contexts...)
    jac2 = jacobian!(f, mysimilar(scen.res1), prep, ba, x, contexts...)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(scen.res1)
        @test mynnz(jac2) == mynnz(scen.res1)
        @test nameof(typeof(jac1)) == nameof(typeof(SMC.sparsity_pattern(prep)))
        @test nameof(typeof(jac2)) == nameof(typeof(SMC.sparsity_pattern(prep)))
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::Scenario{:jacobian,:out,:in})
    @compat (; f, x, y, contexts) = deepcopy(scen)
    f! = f
    prep = prepare_jacobian(f!, mysimilar(y), ba, x, contexts...)

    _, jac1 = value_and_jacobian(f!, mysimilar(y), prep, ba, x, contexts...)
    jac2 = jacobian(f!, mysimilar(y), prep, ba, x, contexts...)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(scen.res1)
        @test mynnz(jac2) == mynnz(scen.res1)
        @test nameof(typeof(jac1)) == nameof(typeof(SMC.sparsity_pattern(prep)))
        @test nameof(typeof(jac2)) == nameof(typeof(SMC.sparsity_pattern(prep)))
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::Scenario{:jacobian,:in,:in})
    @compat (; f, x, y, contexts) = deepcopy(scen)
    f! = f
    prep = prepare_jacobian(f!, mysimilar(y), ba, x, contexts...)

    _, jac1 = value_and_jacobian!(
        f!, mysimilar(y), mysimilar(scen.res1), prep, ba, x, contexts...
    )
    jac2 = jacobian!(f!, mysimilar(y), mysimilar(scen.res1), prep, ba, x, contexts...)

    @testset "Sparsity pattern" begin
        @test mynnz(jac1) == mynnz(scen.res1)
        @test mynnz(jac2) == mynnz(scen.res1)
        @test nameof(typeof(jac1)) == nameof(typeof(SMC.sparsity_pattern(prep)))
        @test nameof(typeof(jac2)) == nameof(typeof(SMC.sparsity_pattern(prep)))
    end
    return nothing
end

## Hessian

function test_sparsity(ba::AbstractADType, scen::Scenario{:hessian,:out,:out})
    @compat (; f, x, y, contexts) = deepcopy(scen)
    prep = prepare_hessian(f, ba, x, contexts...)

    hess1 = hessian(f, prep, ba, x, contexts...)
    _, _, hess2 = value_gradient_and_hessian(f, prep, ba, x, contexts...)

    @testset "Sparsity pattern" begin
        @test mynnz(hess1) == mynnz(scen.res2)
        @test mynnz(hess2) == mynnz(scen.res2)
        @test nameof(typeof(hess1)) == nameof(typeof(SMC.sparsity_pattern(prep)))
        @test nameof(typeof(hess2)) == nameof(typeof(SMC.sparsity_pattern(prep)))
    end
    return nothing
end

function test_sparsity(ba::AbstractADType, scen::Scenario{:hessian,:in,:out})
    @compat (; f, x, y, contexts) = deepcopy(scen)
    prep = prepare_hessian(f, ba, x, contexts...)

    hess1 = hessian!(f, mysimilar(scen.res2), prep, ba, x, contexts...)
    _, _, hess2 = value_gradient_and_hessian!(
        f, mysimilar(x), mysimilar(scen.res2), prep, ba, x, contexts...
    )

    @testset "Sparsity pattern" begin
        @test mynnz(hess1) == mynnz(scen.res2)
        @test mynnz(hess2) == mynnz(scen.res2)
        @test nameof(typeof(hess1)) == nameof(typeof(SMC.sparsity_pattern(prep)))
        @test nameof(typeof(hess2)) == nameof(typeof(SMC.sparsity_pattern(prep)))
    end
    return nothing
end
