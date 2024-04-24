# Table of overloads

This table recaps the features of each extension, with respect to high-level operators.
Each cell can have three values:

- ❌: the backend does not support this operator
- ✅: our extension calls the backend operator and handles preparation if possible
- NA: the operator is not available (e.g. because mutation on scalar function outputs isn't possible).

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

ext_module(ext::Symbol) = Base.get_extension(DifferentiationInterface, ext)

function all_backends_and_extensions()
    return [
        (AutoDiffractor(), ext_module(:DifferentiationInterfaceDiffractorExt)),
        (AutoEnzyme(; mode=Enzyme.Forward), ext_module(:DifferentiationInterfaceEnzymeExt)),
        (AutoEnzyme(; mode=Enzyme.Reverse), ext_module(:DifferentiationInterfaceEnzymeExt)),
        (AutoFastDifferentiation(), ext_module(:DifferentiationInterfaceFastDifferentiationExt),),
        (AutoFiniteDiff(), ext_module(:DifferentiationInterfaceFiniteDiffExt)),
        (AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)),ext_module(:DifferentiationInterfaceFiniteDifferencesExt),),
        (AutoForwardDiff(), ext_module(:DifferentiationInterfaceForwardDiffExt)),
        (AutoPolyesterForwardDiff(; chunksize=1),ext_module(:DifferentiationInterfacePolyesterForwardDiffExt),),
        (AutoReverseDiff(), ext_module(:DifferentiationInterfaceReverseDiffExt)),
        (AutoTapir(), ext_module(:DifferentiationInterfaceTapirExt)),
        (AutoTracker(), ext_module(:DifferentiationInterfaceTrackerExt)),
        (AutoZygote(), ext_module(:DifferentiationInterfaceZygoteExt)),
    ]
end

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
            (:pushforward, (Any, T, Any, Any, Any)),
            (:pushforward!, (Any, Any, T, Any, Any, Any)),
            (:value_and_pushforward, (Any, T, Any, Any, Any)),
            (:value_and_pushforward!, (Any, Any, T, Any, Any, Any)),
        ),
        (
            (:pullback, (Any, T, Any, Any, Any)),
            (:pullback!, (Any, Any, T, Any, Any, Any)),
            (:value_and_pullback, (Any, T, Any, Any, Any)),
            (:value_and_pullback!, (Any, Any, T, Any, Any, Any)),
        ),
    )
end
function operators_and_types_f!(backend::T) where {T<:AbstractADType}
    return (
        # (op,          types_op), 
        # (op!,         types_op!), 
        # (val_and_op,  types_val_and_op),
        # (val_and_op!, types_val_and_op!),
        (
            (:derivative, (Any, Any, T, Any, Any)),
            (:derivative!, (Any, Any, Any, T, Any, Any)),
            (:value_and_derivative, (Any, Any, T, Any, Any)),
            (:value_and_derivative!, (Any, Any, Any, T, Any, Any)),
        ),
        (
            (:gradient, (Any, Any, T, Any, Any)),
            (:gradient!, (Any, Any, Any, T, Any, Any)),
            (:value_and_gradient, (Any, Any, T, Any, Any)),
            (:value_and_gradient!, (Any, Any, Any, T, Any, Any)),
        ),
        (
            (:jacobian, (Any, Any, T, Any, Any)),
            (:jacobian!, (Any, Any, Any, T, Any, Any)),
            (:value_and_jacobian, (Any, Any, T, Any, Any)),
            (:value_and_jacobian!, (Any, Any, Any, T, Any, Any)),
        ),
        (
            (:hessian, (Any, Any, T, Any, Any)),
            (:hessian!, (Any, Any, Any, T, Any, Any)),
            (nothing, nothing),
            (nothing, nothing),
        ),
        (
            (:hvp, (Any, Any, T, Any, Any, Any)),
            (:hvp!, (Any, Any, Any, T, Any, Any, Any)),
            (nothing, nothing),
            (nothing, nothing),
        ),
        (
            (:pushforward, (Any, Any, T, Any, Any, Any)),
            (:pushforward!, (Any, Any, Any, T, Any, Any, Any)),
            (:value_and_pushforward, (Any, Any, T, Any, Any, Any)),
            (:value_and_pushforward!, (Any, Any, Any, T, Any, Any, Any)),
        ),
        (
            (:pullback, (Any, Any, T, Any, Any, Any)),
            (:pullback!, (Any, Any, Any, T, Any, Any, Any)),
            (:value_and_pullback, (Any, Any, T, Any, Any, Any)),
            (:value_and_pullback!, (Any, Any, Any, T, Any, Any, Any)),
        ),
    )
end

function method_overloaded(operator::Symbol, argtypes, m::Module)
    f = @eval DifferentiationInterface.$operator
    ms = methods(f, argtypes, m)

    n = length(ms)
    n == 0 && return "❌"
    n == 1 && return "[✅]($(Base.url(only(ms))))"
    return "✅"
end

io = IOBuffer()

for (backend, ext) in all_backends_and_extensions()
    bname = backend_string(backend)
    btype = typeof(backend)

    # Subsection title
    println(io, "## $bname")

    # First-order table
    println(io, "### Functions `f(x)`")
    println(io)
    println(io, "| Operator | `op` | `op!` | `value_and_op` | `value_and_op!` |")
    println(io, "|:---------|:----:|:-----:|:--------------:|:---------------:|")
    for variants in operators_and_types_f(backend)
        opname = first(first(variants))
        print(io, "| `$opname` |")
        for (op, type_signature) in variants
            if isnothing(op)
                print(io, "NA")
            else
                print(io, method_overloaded(op, type_signature, ext))
            end
            print(io, '|')
        end
        println(io)
    end

    println(io, "### Functions `f!(y, x)`")
    println(io)
    println(io, "| Operator | `op` | `op!` | `value_and_op` | `value_and_op!` |")
    println(io, "|:---------|:----:|:-----:|:--------------:|:---------------:|")
    for variants in operators_and_types_f!(backend)
        opname = first(first(variants))
        print(io, "| `$opname` |")
        for (op, type_signature) in variants
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

overload_tables = Markdown.parse(String(take!(io)))
```

```@example overloads
overload_tables # hide
```