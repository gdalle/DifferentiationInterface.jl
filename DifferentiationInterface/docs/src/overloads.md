# Table of overloads

This table recaps the features of each extension, with respect to high-level operators.
Each cell can have three values:

- ❌: the backend does not support this operator
- ✅: our extension calls the backend operator and handles preparation if possible
- NA: the operator is not available

Checkmarks (✅) are clickable and link to the source code.

```@setup overloads
using ADTypes
using DifferentiationInterface
using DifferentiationInterface: backend_string
using Markdown: Markdown
using Diffractor: Diffractor
using Enzyme: Enzyme
using FastDifferentiation: FastDifferentiation
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Tapir: Tapir
using Tracker: Tracker
using Zygote: Zygote

function operators_and_types_f(backend::T) where {T<:AbstractADType}
    return (
        # (op,          types_op), 
        # (op!,         types_op!), 
        # (val_and_op,  types_val_and_op),
        # (val_and_op!, types_val_and_op!),
        (
            (:derivative, (Any, T, Any, Any)),
            (:derivative!, (Any, Any, T, Any, Any)),
            (:value_and_derivative, (Any, T, Any, Any)),
            (:value_and_derivative!, (Any, Any, T, Any, Any)),
        ),
        (
            (:gradient, (Any, T, Any, Any)),
            (:gradient!, (Any, Any, T, Any, Any)),
            (:value_and_gradient, (Any, T, Any, Any)),
            (:value_and_gradient!, (Any, Any, T, Any, Any)),
        ),
        (
            (:jacobian, (Any, T, Any, Any)),
            (:jacobian!, (Any, Any, T, Any, Any)),
            (:value_and_jacobian, (Any, T, Any, Any)),
            (:value_and_jacobian!, (Any, Any, T, Any, Any)),
        ),
        (
            (:hessian, (Any, T, Any, Any)),
            (:hessian!, (Any, Any, T, Any, Any)),
            (nothing, nothing),
            (nothing, nothing),
        ),
        (
            (:hvp, (Any, T, Any, Any, Any)),
            (:hvp!, (Any, Any, T, Any, Any, Any)),
            (nothing, nothing),
            (nothing, nothing),
        ),
        (
            (:pullback, (Any, T, Any, Any, Any)),
            (:pullback!, (Any, Any, T, Any, Any, Any)),
            (:value_and_pullback, (Any, T, Any, Any, Any)),
            (:value_and_pullback!, (Any, Any, T, Any, Any, Any)),
        ),
        (
            (:pushforward, (Any, T, Any, Any, Any)),
            (:pushforward!, (Any, Any, T, Any, Any, Any)),
            (:value_and_pushforward, (Any, T, Any, Any, Any)),
            (:value_and_pushforward!, (Any, Any, T, Any, Any, Any)),
        ),
    )
end
function operators_and_types_f!(backend::T) where {T<:AbstractADType}
    return (
        (
            (:derivative, (Any, Any, T, Any, Any)),
            (:derivative!, (Any, Any, Any, T, Any, Any)),
            (:value_and_derivative, (Any, Any, T, Any, Any)),
            (:value_and_derivative!, (Any, Any, Any, T, Any, Any)),
        ),
        (
            (:jacobian, (Any, Any, T, Any, Any)),
            (:jacobian!, (Any, Any, Any, T, Any, Any)),
            (:value_and_jacobian, (Any, Any, T, Any, Any)),
            (:value_and_jacobian!, (Any, Any, Any, T, Any, Any)),
        ),
        (
            (:pullback, (Any, Any, T, Any, Any, Any)),
            (:pullback!, (Any, Any, Any, T, Any, Any, Any)),
            (:value_and_pullback, (Any, Any, T, Any, Any, Any)),
            (:value_and_pullback!, (Any, Any, Any, T, Any, Any, Any)),
        ),
        (
            (:pushforward, (Any, Any, T, Any, Any, Any)),
            (:pushforward!, (Any, Any, Any, T, Any, Any, Any)),
            (:value_and_pushforward, (Any, Any, T, Any, Any, Any)),
            (:value_and_pushforward!, (Any, Any, Any, T, Any, Any, Any)),
        ),
    )
end

function method_overloaded(operator::Symbol, argtypes, ext::Module)
    f = @eval DifferentiationInterface.$operator
    ms = methods(f, argtypes, ext)

    n = length(ms)
    n == 0 && return "❌"
    n == 1 && return "[✅]($(Base.url(only(ms))))"
    return "[✅]($(Base.url(first(ms))))" # Optional TODO: return all URLs?
end

function print_overload_table(io::IO, operators_and_types, ext::Module)
    println(io, "| Operator | `op` | `op!` | `value_and_op` | `value_and_op!` |")
    println(io, "|:---------|:----:|:-----:|:--------------:|:---------------:|")
    for operator_variants in operators_and_types
        opname = first(first(operator_variants))
        print(io, "| `$opname` |")
        for (op, type_signature) in operator_variants
            if isnothing(op)
                print(io, "NA")
            else
                print(io, method_overloaded(op, type_signature, ext))
            end
            print(io, '|')
        end
        println(io)
    end
end

function print_overloads(backend, ext::Symbol)
    io = IOBuffer()
    ext = Base.get_extension(DifferentiationInterface, ext)

    println(io, "### `f(x)`")
    println(io)
    print_overload_table(io, operators_and_types_f(backend), ext)

    println(io, "### `f!(y, x)`")
    println(io)
    print_overload_table(io, operators_and_types_f!(backend), ext)

    return Markdown.parse(String(take!(io)))
end
```

## Diffractor (forward/reverse)
```@example overloads
print_overloads(AutoDiffractor(), :DifferentiationInterfaceDiffractorExt) # hide
```

## Enzyme (forward)
```@example overloads
print_overloads(AutoEnzyme(; mode=Enzyme.Forward), :DifferentiationInterfaceEnzymeExt) # hide
```

## Enzyme (reverse)
```@example overloads
print_overloads(AutoEnzyme(; mode=Enzyme.Reverse), :DifferentiationInterfaceEnzymeExt) # hide
```

## FastDifferentiation (symbolic)
```@example overloads
print_overloads(AutoFastDifferentiation(), :DifferentiationInterfaceFastDifferentiationExt) # hide
```

## FiniteDiff (forward)
```@example overloads
print_overloads(AutoFiniteDiff(), :DifferentiationInterfaceFiniteDiffExt) # hide
```

## FiniteDifferences (forward)
```@example overloads
print_overloads(AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)), :DifferentiationInterfaceFiniteDifferencesExt) # hide
```

## ForwardDiff (forward)
```@example overloads
print_overloads(AutoForwardDiff(), :DifferentiationInterfaceForwardDiffExt) # hide
```

## PolyesterForwardDiff (forward)
```@example overloads
print_overloads(AutoPolyesterForwardDiff(; chunksize=1), :DifferentiationInterfacePolyesterForwardDiffExt) # hide
```

## ReverseDiff (reverse)
```@example overloads
print_overloads(AutoReverseDiff(), :DifferentiationInterfaceReverseDiffExt) # hide
```

## Tapir (reverse)
```@example overloads
print_overloads(AutoTapir(), :DifferentiationInterfaceTapirExt) # hide
```

## Tracker (reverse)
```@example overloads
print_overloads(AutoTracker(), :DifferentiationInterfaceTrackerExt) # hide
```

## Zygote (reverse)
```@example overloads
print_overloads(AutoZygote(), :DifferentiationInterfaceZygoteExt) # hide
```
