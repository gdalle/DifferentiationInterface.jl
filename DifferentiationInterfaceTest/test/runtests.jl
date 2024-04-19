include("test_imports.jl")

## Main tests

@testset verbose = true "DifferentiationInterfaceTest.jl" begin
    @testset verbose = true "Formal tests" begin
        @testset "Aqua" begin
            Aqua.test_all(
                DifferentiationInterfaceTest;
                ambiguities=false,
                deps_compat=(check_extras = false),
                persistent_tasks=false,
                stale_deps=false,
            )
        end
        @testset "JuliaFormatter" begin
            @test JuliaFormatter.format(
                DifferentiationInterfaceTest; verbose=false, overwrite=false
            )
        end
        @testset verbose = true "JET" begin
            JET.test_package(DifferentiationInterfaceTest; target_defined_modules=true)
        end
    end

    @testset verbose = false "Zero backends" begin
        include("zero_backends.jl")
    end

    @testset verbose = false "ForwardDiff" begin
        include("forwarddiff.jl")
    end

    @testset verbose = false "Symbolics" begin
        include("symbolics.jl")
    end
end;
