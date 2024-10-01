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
    for category in readdir(@__DIR__)
        isdir(category) || continue
        for folder in readdir(joinpath(@__DIR__, category))
            if GROUP == "All" || (category, folder) == split(GROUP, '/')
                @testset verbose = true "$category/$folder" begin
                    @testset "$file" for file in
                                         readdir(joinpath(@__DIR__, category, folder))
                        @info "Testing $category/$folder/$file"
                        include(joinpath(@__DIR__, category, folder, file))
                    end
                end
            end
        end
    end
end;
