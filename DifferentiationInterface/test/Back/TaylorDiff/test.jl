using Pkg
Pkg.add("TaylorDiff")

using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using TaylorDiff: TaylorDiff
using Test

LOGGING = get(ENV, "CI", "false") == "false"

backends = [AutoTaylorDiff(), AutoTaylorDiff(; order=2)]

for backend in backends
    @test check_available(backend)
    @test check_inplace(backend)
end
