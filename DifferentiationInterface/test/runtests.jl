using DifferentiationInterface
using Pkg
using Test

DIT_PATH = joinpath(@__DIR__, "..", "..", "DifferentiationInterfaceTest")
if isdir(DIT_PATH)
    Pkg.develop(; path=DIT_PATH)
else
    Pkg.add("DifferentiationInterfaceTest")
end

LOGGING = get(ENV, "CI", "false") == "false"

GROUP = get(ENV, "JULIA_DI_TEST_GROUP", "All")

ALL_BACKENDS = [
    "ChainRulesCore",
    "Diffractor",
    "Enzyme",
    "FiniteDiff",
    "FiniteDifferences",
    "FastDifferentiation",
    "ForwardDiff",
    "PolyesterForwardDiff",
    "ReverseDiff",
    "Symbolics",
    "Tapir",
    "Tracker",
    "Zygote",
]

## Main tests

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
        Pkg.add(ALL_BACKENDS)
        @testset verbose = true "$folder" for folder in readdir(joinpath(@__DIR__, "Back"))
            @testset "$file" for file in readdir(joinpath(@__DIR__, "Back", folder))
                @info "Testing Back/$folder/$file"
                include(joinpath(@__DIR__, "Back", folder, file))
            end
        end
    elseif startswith(GROUP, "Back") || startswith(GROUP, "Down")
        category, folder = split(GROUP, '/')
        backends = split(folder, '-')
        Pkg.add(backends)
        @testset verbose = true "$category/$folder" begin
            @testset "$file" for file in readdir(joinpath(@__DIR__, category, folder))
                @info "Testing $category/$folder/$file"
                include(joinpath(@__DIR__, category, folder, file))
            end
        end
    end
end;
