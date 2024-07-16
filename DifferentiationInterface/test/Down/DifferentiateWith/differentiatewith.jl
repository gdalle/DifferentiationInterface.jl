using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterfaceTest as DIT
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Test

function zygote_breaking_scenarios()
    onearg_scens = filter(default_scenarios()) do scen
        DIT.nb_args(scen) == 1
    end
    bad_onearg_scens = map(onearg_scens) do scen
        function bad_f(x)
            a = Vector{eltype(x)}(undef, 1)
            a[1] = sum(x)
            return scen.f(x)
        end
        wrapped_bad_f = DifferentiateWith(bad_f, AutoForwardDiff())
        bad_scen = DIT.change_function(scen, wrapped_bad_f)
        return bad_scen
    end
    return bad_onearg_scens
end

test_differentiation(
    AutoZygote(), zygote_breaking_scenarios(); second_order=false, logging=LOGGING
)
