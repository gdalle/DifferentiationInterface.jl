module DifferentiationInterfaceTaylorDiffExt

using ADTypes: AutoTaylorDiff
import DifferentiationInterface as DI
using TaylorDiff

DI.check_available(::AutoTaylorDiff) = true
end
