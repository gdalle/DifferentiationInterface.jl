# Implementations

DifferentiationInterface.jl provides a handful of [operators](@ref "Operators") like [`gradient`](@ref) or [`jacobian`](@ref), each with several variants: out-of-place or in-place behavior, with or without primal output value.

While it is possible to define every operator using just [`pushforward`](@ref) and [`pullback`](@ref), some backends have more efficient implementations of high-level operators.
When they are available, we nearly always call these backend-specific overloads.
We also adapt the preparation phase accordingly.
This page gives details on each backend's bindings.

The tables below summarize all implemented overloads for each backend.
The cells can have three values:

- ❌: the operator is not overloaded because the backend does not support it
- ✅: the operator is overloaded
- NA: the operator does not exist

!!! tip
    Check marks (✅) are clickable and link to the source code.

```@setup overloads
using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface: inplace_support, InPlaceSupported
using Markdown: Markdown

using Diffractor: Diffractor
using Enzyme: Enzyme
using FastDifferentiation: FastDifferentiation
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Symbolics: Symbolics
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
            (:derivative, (Any, Any, T, Any)),
            (:derivative!, (Any, Any, Any, T, Any)),
            (:value_and_derivative, (Any, Any, T, Any)),
            (:value_and_derivative!, (Any, Any, Any, T, Any)),
        ),
        (
            (:gradient, (Any, Any, T, Any)),
            (:gradient!, (Any, Any, Any, T, Any)),
            (:value_and_gradient, (Any, Any, T, Any)),
            (:value_and_gradient!, (Any, Any, Any, T, Any)),
        ),
        (
            (:jacobian, (Any, Any, T, Any)),
            (:jacobian!, (Any, Any, Any, T, Any)),
            (:value_and_jacobian, (Any, Any, T, Any)),
            (:value_and_jacobian!, (Any, Any, Any, T, Any)),
        ),
        (
            (:hessian, (Any, Any, T, Any)),
            (:hessian!, (Any, Any, Any, T, Any)),
            (nothing, nothing),
            (nothing, nothing),
        ),
        (
            (:hvp, (Any, Any, T, Any, Any)),
            (:hvp!, (Any, Any, Any, T, Any, Any)),
            (nothing, nothing),
            (nothing, nothing),
        ),
        (
            (:pullback, (Any, Any, T, Any, Any)),
            (:pullback!, (Any, Any, Any, T, Any, Any)),
            (:value_and_pullback, (Any, Any, T, Any, Any)),
            (:value_and_pullback!, (Any, Any, Any, T, Any, Any)),
        ),
        (
            (:pushforward, (Any, Any, T, Any, Any)),
            (:pushforward!, (Any, Any, Any, T, Any, Any)),
            (:value_and_pushforward, (Any, Any, T, Any, Any)),
            (:value_and_pushforward!, (Any, Any, Any, T, Any, Any)),
        ),
    )
end

function operators_and_types_f!(backend::T) where {T<:AbstractADType}
    return (
        (
            (:derivative, (Any, Any, Any, T, Any)),
            (:derivative!, (Any, Any, Any, Any, T, Any)),
            (:value_and_derivative, (Any, Any, Any, T, Any)),
            (:value_and_derivative!, (Any, Any, Any, Any, T, Any)),
        ),
        (
            (:jacobian, (Any, Any, Any, T, Any)),
            (:jacobian!, (Any, Any, Any, Any, T, Any)),
            (:value_and_jacobian, (Any, Any, Any, T, Any)),
            (:value_and_jacobian!, (Any, Any, Any, Any, T, Any)),
        ),
        (
            (:pullback, (Any, Any, Any, T, Any, Any)),
            (:pullback!, (Any, Any, Any, Any, T, Any, Any)),
            (:value_and_pullback, (Any, Any, Any, T, Any, Any)),
            (:value_and_pullback!, (Any, Any, Any, Any, T, Any, Any)),
        ),
        (
            (:pushforward, (Any, Any, Any, T, Any, Any)),
            (:pushforward!, (Any, Any, Any, Any, T, Any, Any)),
            (:value_and_pushforward, (Any, Any, Any, T, Any, Any)),
            (:value_and_pushforward!, (Any, Any, Any, Any, T, Any, Any)),
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

    println(io, "#### Out-of-place functions `f(x) = y`")
    println(io)
    print_overload_table(io, operators_and_types_f(backend), ext)

    println(io, "#### In-place functions `f!(y, x) = nothing`")
    println(io)
    if inplace_support(backend) == InPlaceSupported()
        print_overload_table(io, operators_and_types_f!(backend), ext)
    else
        println(io, "Backend doesn't support in-place functions.")
    end

    return Markdown.parse(String(take!(io)))
end
```

## ChainRulesCore

For [`pullback`](@ref), same-point preparation runs the forward sweep and returns the pullback closure.

## Diffractor

```@example overloads
print_overloads(AutoDiffractor(), :DifferentiationInterfaceDiffractorExt) # hide
```

## Enzyme

### Forward mode

In forward mode, for [`gradient`](@ref) and [`jacobian`](@ref), preparation chooses a number of chunks.

```@example overloads
print_overloads(AutoEnzyme(; mode=Enzyme.Forward), :DifferentiationInterfaceEnzymeExt) # hide
```

### Reverse mode

```@example overloads
print_overloads(AutoEnzyme(; mode=Enzyme.Reverse), :DifferentiationInterfaceEnzymeExt) # hide
```

## FastDifferentiation

Preparation generates an [executable function](https://brianguenter.github.io/FastDifferentiation.jl/stable/makefunction/) from the symbolic expression of the differentiated function.

!!! warning
    Preparation can be very slow for symbolic AD.

```@example overloads
print_overloads(AutoFastDifferentiation(), :DifferentiationInterfaceFastDifferentiationExt) # hide
```

## FiniteDiff

Whenever possible, preparation creates a cache object.

```@example overloads
print_overloads(AutoFiniteDiff(), :DifferentiationInterfaceFiniteDiffExt) # hide
```

## FiniteDifferences

```@example overloads
print_overloads(AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)), :DifferentiationInterfaceFiniteDifferencesExt) # hide
```

## ForwardDiff

Wherever possible, preparation creates a [config](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#Preallocating/Configuring-Work-Buffers).
For [`pushforward`](@ref), preparation allocates the necessary space for `Dual` number computations.

```@example overloads
print_overloads(AutoForwardDiff(), :DifferentiationInterfaceForwardDiffExt) # hide
```

## PolyesterForwardDiff

```@example overloads
print_overloads(AutoPolyesterForwardDiff(; chunksize=1), :DifferentiationInterfacePolyesterForwardDiffExt) # hide
```

## ReverseDiff

Wherever possible, preparation records a [tape](https://juliadiff.org/ReverseDiff.jl/dev/api/#The-AbstractTape-API) of the function's execution.

!!! warning
    This tape is specific to the control flow inside the function, and cannot be reused if the control flow is value-dependent (like `if x[1] > 0`).

```@example overloads
print_overloads(AutoReverseDiff(), :DifferentiationInterfaceReverseDiffExt) # hide
```

## Symbolics

Preparation generates an [executable function](https://docs.sciml.ai/Symbolics/stable/manual/build_function/) from the symbolic expression of the differentiated function.

!!! warning
    Preparation can be very slow for symbolic AD.

```@example overloads
print_overloads(AutoSymbolics(), :DifferentiationInterfaceSymbolicsExt) # hide
```

## Tapir

For [`pullback`](@ref), preparation [builds the reverse rule](https://github.com/withbayes/Tapir.jl?tab=readme-ov-file#how-it-works) of the function.

```@example overloads
print_overloads(AutoTapir(), :DifferentiationInterfaceTapirExt) # hide
```

## Tracker

For [`pullback`](@ref), same-point preparation runs the forward sweep and returns the pullback closure at `x`.

```@example overloads
print_overloads(AutoTracker(), :DifferentiationInterfaceTrackerExt) # hide
```

## Zygote

For [`pullback`](@ref), same-point preparation runs the forward sweep and returns the pullback closure at `x`.

```@example overloads
print_overloads(AutoZygote(), :DifferentiationInterfaceZygoteExt) # hide
```
