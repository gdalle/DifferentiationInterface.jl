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
        @testset verbose = true "$folder" for folder in ("Backends",)
            @testset verbose = true "$subfolder" for subfolder in
                                                     readdir(joinpath(@__DIR__, folder))
                @testset "$file" for file in readdir(joinpath(@__DIR__, folder, subfolder))
                    @info "Testing $folder/$subfolder/$file"
                    include(joinpath(@__DIR__, folder, subfolder, file))
                end
            end
        end
    elseif startswith(GROUP, "Backends")
        @testset verbose = true "$GROUP" begin
            backends_str = split(GROUP, '/')[2]
            backends = split(backends_str, '-')
            Pkg.add(backends)
            @testset "$file" for file in
                                 readdir(joinpath(@__DIR__, "Backends", backends_str))
                @info "Testing Backends/$backends_str/$file"
                include(joinpath(@__DIR__, "Backends", backends_str, file))
            end
        end
    end
end;
