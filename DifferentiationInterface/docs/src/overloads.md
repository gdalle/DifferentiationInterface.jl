# Table of overloads

This table recaps the features of each extension, with respect to high-level operators.
Each cell can have three values

- ❌: the backend does not support this operator
- ✅: our extension calls the backend operator and handles preparation if possible
- NA: The operator is not available (e.g. because mutation on scalar function outputs isn't possible).

Checkmarks (✅) are clickable and link to the source code.

```@setup overloads
using ADTypes
using DifferentiationInterface
using DifferentiationInterface: backend_string
import Markdown
import Diffractor, Enzyme, FastDifferentiation, FiniteDiff, FiniteDifferences, ForwardDiff, PolyesterForwardDiff, ReverseDiff, Tapir, Tracker, Zygote

ext_module(ext::Symbol) = Base.get_extension(DifferentiationInterface, ext)

function all_backends_and_extensions()
    return [
        (AutoDiffractor(), ext_module(:DifferentiationInterfaceDiffractorExt)),
        (AutoEnzyme(; mode=Enzyme.Forward), ext_module(:DifferentiationInterfaceEnzymeExt)),
        (AutoEnzyme(; mode=Enzyme.Reverse), ext_module(:DifferentiationInterfaceEnzymeExt)),
        (AutoFastDifferentiation(), ext_module(:DifferentiationInterfaceFastDifferentiationExt)),
        (AutoFiniteDiff(), ext_module(:DifferentiationInterfaceFiniteDiffExt)),
        (AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(3, 1)), ext_module(:DifferentiationInterfaceFiniteDifferencesExt)),
        (AutoForwardDiff(), ext_module(:DifferentiationInterfaceForwardDiffExt)),
        (AutoPolyesterForwardDiff(; chunksize=1), ext_module(:DifferentiationInterfacePolyesterForwardDiffExt)),
        (AutoReverseDiff(), ext_module(:DifferentiationInterfaceReverseDiffExt)),
        (AutoTapir(), ext_module(:DifferentiationInterfaceTapirExt)),
        (AutoTracker(), ext_module(:DifferentiationInterfaceTrackerExt)),
        (AutoZygote(), ext_module(:DifferentiationInterfaceZygoteExt)),
    ]
end

function operators(backend::T) where {T<:AbstractADType} 
    return (
        # (operator, signature_f, signature_f!)
        (:value_and_derivative!, (Any, Any, T, Any, Any), (Any, Any, Any, T, Any, Any)), 
        (:value_and_derivative, (Any, T, Any, Any), (Any, Any, T, Any, Any)), 
        (:derivative!, (Any, Any, T, Any, Any), (Any, Any, Any, T, Any, Any)), 
        (:derivative, (Any, T, Any, Any), (Any, Any, T, Any, Any)),   
        (:value_and_gradient!, (Any, Any, T, Any, Any), nothing), 
        (:value_and_gradient, (Any, T, Any, Any), nothing), 
        (:gradient!, (Any, Any, T, Any, Any), nothing), 
        (:gradient, (Any, T, Any, Any), nothing), 
        (:value_and_jacobian!, (Any, Any, T, Any, Any), (Any, Any, Any, T, Any, Any)), 
        (:value_and_jacobian, (Any, T, Any, Any), (Any, Any, T, Any, Any)), 
        (:jacobian!, (Any, Any, T, Any, Any), (Any, Any, Any, T, Any, Any)), 
        (:jacobian, (Any, T, Any, Any), (Any, Any, T, Any, Any)), 
        (:hvp!, (Any, Any, T, Any, Any, Any), nothing), 
        (:hvp, (Any, T, Any, Any, Any), nothing), 
        (:hessian!, (Any, Any, T, Any, Any), nothing),
        (:hessian, (Any, T, Any, Any), nothing), 
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
    println(io, "| Operator | `f(x)` | `f!(y, x)` |")
    println(io, "|:---------|:------:|:----------:|")
    for (op, signature_f, signature_f!) in operators(backend)
        # First column: f(x)
        print(io, "| `$op` |", method_overloaded(op, signature_f, ext), '|') 
        # Second column: f!(y, x)
        if isnothing(signature_f!)
            println(io, "NA |")
        else
            println(io, method_overloaded(op, signature_f!, ext), '|')
        end
    end
end

overload_tables = Markdown.parse(String(take!(io)))
```

```@example overloads
overload_tables # hide
```