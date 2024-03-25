include("test_imports.jl")

using Zygote: Zygote

@test check_available(AutoZygote())
@test !check_mutation(AutoZygote())
@test_skip !check_hessian(AutoZygote())

test_differentiation(AutoZygote(); second_order=false);

test_differentiation(
    AutoZygote(),
    all_operators(),
    weird_array_scenarios(; static=true, component=true, gpu=true);
    second_order=false,
);
