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

## Main tests

@testset verbose = true "DifferentiationInterface.jl" begin
    if GROUP == "Formalities" || GROUP == "All"
        @testset "Formalities/$file" for file in readdir(joinpath(@__DIR__, "Formalities"))
            @info "Testing Formalities/$file)"
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
        Pkg.add([
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
        ])
        @testset verbose = true "$folder" for folder in ("Single", "Double")
            files = filter(f -> endswith(f, ".jl"), readdir(joinpath(@__DIR__, folder)))
            @testset "$file" for file in files
                @info "Testing $folder/$file"
                include(joinpath(@__DIR__, folder, file))
            end
        end
    elseif startswith(GROUP, "Single")
        b1 = split(GROUP, '/')[2]
        @testset "Single/$b1" begin
            @info "Testing Single/$b1"
            Pkg.add(b1)
            include(joinpath(@__DIR__, "Single", "$b1.jl"))
        end
    elseif startswith(GROUP, "Double")
        b1, b2 = split(split(GROUP, '/')[2], '-')
        @testset "Single/$b1-$b2" begin
            @info "Testing Double/$b1-$b2"
            Pkg.add([b1, b2])
            include(joinpath(@__DIR__, "Double", "$b1-$b2.jl"))
        end
    end
end;
