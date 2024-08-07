using DifferentiationInterface
using Pkg
using Test

DIT_PATH = joinpath(@__DIR__, "..", "..", "DifferentiationInterfaceTest")
if isdir(DIT_PATH)
    Pkg.develop(; path=DIT_PATH)
else
    Pkg.add("DifferentiationInterfaceTest")
end

#GROUP = get(ENV, "JULIA_DI_TEST_GROUP", "All")
GROUP = get(ENV, "JULIA_DI_TEST_GROUP", "Back/GTPSA")

## Main tests

function subtest(category, folder)
    @testset "$file" for file in readdir(joinpath(@__DIR__, category, folder))
        @info "Testing $category/$folder/$file"
        include(joinpath(@__DIR__, category, folder, file))
    end
    return nothing
end

@testset verbose = true "DifferentiationInterface.jl" begin
    if GROUP == "Formalities" || GROUP == "All"
        @testset "Formalities/$file" for file in readdir(joinpath(@__DIR__, "Formalities"))
            @info "Testing Formalities/$file"
            include(joinpath(@__DIR__, "Formalities", file))
        end
    end

    if GROUP == "Internals" || GROUP == "All"
        @testset "Internals/$file" for file in readdir(joinpath(@__DIR__, "Internals"))
            @info "Testing Internals/$file"
            include(joinpath(@__DIR__, "Internals", file))
        end
    end

    if GROUP == "All"
        @testset verbose = true "$category" for category in ["Back", "Down"]
            @testset verbose = true "$folder" for folder in
                                                  readdir(joinpath(@__DIR__, category))
                subtest(category, folder)
            end
        end
    elseif startswith(GROUP, "Back") || startswith(GROUP, "Down")
        category, folder = split(GROUP, '/')
        @testset verbose = true "$category" begin
            @testset verbose = true "$folder" begin
                subtest(category, folder)
            end
        end
    end
end;
