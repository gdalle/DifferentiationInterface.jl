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

function subtest(category, folder; fake=false)
    @testset verbose = true "$category/$folder" begin
        @testset "$file" for file in readdir(joinpath(@__DIR__, category, folder))
            @info "Testing $category/$folder/$file"
            if !fake
                include(joinpath(@__DIR__, category, folder, file))
            end
        end
    end
end

@testset verbose = true "DifferentiationInterface.jl" begin
    if GROUP == "All"
        for category in readdir(@__DIR__)
            isdir(category) || continue
            for folder in readdir(joinpath(@__DIR__, category))
                subtest(category, folder; fake=true)
            end
        end
    else
        category, folder = split(GROUP, '/')
        subtest(category, folder)
    end
end;
