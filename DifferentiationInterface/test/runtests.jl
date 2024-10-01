using DifferentiationInterface
using Pkg
using Test

DIT_PATH = joinpath(@__DIR__, "..", "..", "DifferentiationInterfaceTest")
if isdir(DIT_PATH)
    Pkg.develop(; path=DIT_PATH)
else
    Pkg.add("DifferentiationInterfaceTest")
end

GROUP = get(ENV, "JULIA_DI_TEST_GROUP", "All")

## Main tests

@testset verbose = true "DifferentiationInterface.jl" begin
    if GROUP == "All"
        @testset verbose = true for category in readdir(@__DIR__)
            isdir(joinpath(@__DIR__, category)) || continue
            @testset verbose = true for folder in readdir(joinpath(@__DIR__, category))
                isdir(joinpath(@__DIR__, category, folder)) || continue
                @testset "$file" for file in readdir(joinpath(@__DIR__, category, folder))
                    endswith(file, ".jl") || continue
                    @info "Testing $category/$folder/$file"
                    include(joinpath(@__DIR__, category, folder, file))
                end
            end
        end
    else
        category, folder = split(GROUP, '/')
        @testset verbose = true "$category" begin
            @testset verbose = true "$folder" begin
                @testset "$file" for file in readdir(joinpath(@__DIR__, category, folder))
                    endswith(file, ".jl") || continue
                    @info "Testing $category/$folder/$file"
                    include(joinpath(@__DIR__, category, folder, file))
                end
            end
        end
    end
end;
