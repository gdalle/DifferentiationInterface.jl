include("test_imports.jl")

## Main tests

@testset verbose = true "DifferentiationInterface.jl" begin
    @testset verbose = true "Formal tests" begin
        @testset "Aqua" begin
            Aqua.test_all(
                DifferentiationInterface;
                ambiguities=false,
                deps_compat=(check_extras = false),
            )
        end
        @testset "JuliaFormatter" begin
            @test JuliaFormatter.format(
                DifferentiationInterface; verbose=false, overwrite=false
            )
        end
        @testset "JET" begin
            JET.test_package(
                DifferentiationInterface; target_modules=(DifferentiationInterface,)
            )
            JET.test_package(
                DifferentiationInterfaceTest; target_modules=(DifferentiationInterfaceTest,)
            )
        end
    end

    Documenter.doctest(DifferentiationInterface)

    @testset "Zero backends" begin
        include("zero_backends.jl")
    end

    @testset verbose = true "First order" begin
        include("first_order.jl")
    end

    @testset verbose = true "Second order" begin
        include("second_order.jl")
    end

    @testset verbose = true "Bonus round" begin
        @testset "Type stability" begin
            include("type_stability.jl")
        end

        @testset "Weird arrays" begin
            include("weird_arrays.jl")
        end
    end
end;
