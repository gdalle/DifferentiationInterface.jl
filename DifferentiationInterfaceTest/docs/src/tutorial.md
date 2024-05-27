# Tutorial

We present a typical workflow with DifferentiationInterfaceTest.jl, building on the tutorial of the [DifferentiationInterface.jl documentation](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface) (which we encourage you to read first).

```@repl tuto
using DifferentiationInterface, DifferentiationInterfaceTest
import ForwardDiff, Enzyme
import DataFrames, Markdown, PrettyTables, Printf
```

## Introduction

The AD backends we want to compare are [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).

```@repl tuto
backends = [AutoForwardDiff(), AutoEnzyme(; mode=Enzyme.Reverse)]
```

To do that, we are going to take gradients of a simple function:

```@repl tuto
f(x::AbstractArray) = sum(sin, x)
```

Of course we know the true gradient mapping:

```@repl tuto
∇f(x::AbstractArray) = cos.(x)
```

DifferentiationInterfaceTest.jl relies with so-called "scenarios", in which you encapsulate the information needed for your test:

- the function `f`
- the input `x` (and output `y` for mutating functions)
- optionally a reference `ref` to check against

There is one scenario per operator, and so here we will use [`GradientScenario`](@ref):

```@repl tuto
scenarios = [
    GradientScenario(f; x=rand(Float32, 3), ref=∇f, place=:inplace),
    GradientScenario(f; x=rand(Float64, 3, 2), ref=∇f, place=:inplace)
];
```

## Testing

The main entry point for testing is the function [`test_differentiation`](@ref).
It has many options, but the main ingredients are the following:

```@repl tuto
test_differentiation(
    backends,  # the backends you want to compare
    scenarios,  # the scenarios you defined,
    correctness=true,  # compares values against the reference
    type_stability=false,  # checks type stability with JET.jl
    detailed=true,  # prints a detailed test set
)
```

If you are too lazy to manually specify the reference, you can also provide an AD backend as the `ref_backend` keyword argument, which will serve as the ground truth for comparison.

## Benchmarking

Once you are confident that your backends give the correct answers, you probably want to compare their performance.
This is made easy by the [`benchmark_differentiation`](@ref) function, whose syntax should feel familiar:

```@repl tuto
benchmark_result = benchmark_differentiation(backends, scenarios);
```

The resulting object is a `Vector` of [`DifferentiationBenchmarkDataRow`](@ref), which can easily be converted into a `DataFrame` from [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl):

```@repl tuto
df = DataFrames.DataFrame(benchmark_result)
```

Here's what the resulting `DataFrame` looks like with all its columns.
Note that we only compare (possibly) in-place operators, because they are always more efficient.

```@example tuto
table = PrettyTables.pretty_table(
    String,
    df;
    backend=Val(:markdown),
    header=names(df),
)

Markdown.parse(table)
```
