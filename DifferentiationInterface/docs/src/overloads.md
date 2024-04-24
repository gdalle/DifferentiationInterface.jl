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

operators = (
    :value_and_derivative!, 
    :value_and_derivative, 
    :derivative!, 
    :derivative,   
    :value_and_gradient!, 
    :value_and_gradient, 
    :gradient!, 
    :gradient, 
    :value_and_jacobian!, 
    :value_and_jacobian, 
    :jacobian!, 
    :jacobian, 
    :hvp!, 
    :hvp, 
    :hessian!,
    :hessian, 
)

isinplace(op::Symbol) = contains(string(op), '!')

function supports_inplace_fn(op::Symbol)
    name = string(op)
    contains(name, "gradient") && return false
    contains(name, "hvp") && return false
    contains(name, "hessian") && return false
    return true
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
    for op in operators
        print(io, "| `$op` | ")
        # Column f(x)
        if isinplace(op)
            print(io, method_overloaded(op, (Any, Any, btype, Any, Any), ext))
        else
            print(io, method_overloaded(op, (Any, btype, Any, Any), ext))
        end
        print(io, '|')
        # Column f!(x)
        if !supports_inplace_fn(op)
            print(io, "NA")
        else
            if isinplace(op)
                print(io, method_overloaded(op, (Any, Any, Any, btype, Any, Any), ext))
            else
                print(io, method_overloaded(op, (Any, Any, btype, Any, Any), ext))
            end
        end
        println(io, '|')
    end
end

overload_tables = Markdown.parse(String(take!(io)))
```

```@example overloads
overload_tables # hide
```