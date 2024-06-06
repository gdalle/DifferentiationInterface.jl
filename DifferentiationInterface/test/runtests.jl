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
        @testset verbose = true "$folder" for folder in ("Single", "Double")
            @testset verbose = true "$subfolder" for subfolder in
                                                     readdir(joinpath(@__DIR__, folder))
                @testset "$file" for file in readdir(joinpath(@__DIR__, folder, subfolder))
                    @info "Testing $folder/$subfolder/$file"
                    include(joinpath(@__DIR__, folder, subfolder, file))
                end
            end
        end
    elseif startswith(GROUP, "Single")
        b1 = split(GROUP, '/')[2]
        @testset verbose = true "Single/$b1" begin
            Pkg.add(b1)
            @testset "$file" for file in readdir(joinpath(@__DIR__, "Single", "$b1"))
                @info "Testing Single/$b1/$file"
                include(joinpath(@__DIR__, "Single", "$b1", file))
            end
        end
    elseif startswith(GROUP, "Double")
        b1, b2 = split(split(GROUP, '/')[2], '-')
        @testset verbose = true "Double/$b1-$b2" begin
            Pkg.add([b1, b2])
            @testset "$file" for file in readdir(joinpath(@__DIR__, "Double", "$b1-$b2"))
                @info "Testing Double/$b1-$b2/$file"
                include(joinpath(@__DIR__, "Double", "$b1-$b2", file))
            end
        end
    end
end;
