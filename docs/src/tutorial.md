# Tutorial

We present a typical workflow with DifferentiationInterface.jl and showcase its potential performance benefits.

```@repl tuto
using ADTypes, BenchmarkTools, DifferentiationInterface
import ForwardDiff, Enzyme, DataFrames
```

## Computing a gradient

A common use case of Automatic Differentiation (AD) is optimizing real-valued functions with first- or second-order methods.
Let's define a simple objective

```@repl tuto
f(x::AbstractArray) = sum(abs2, x)
```

and a random input vector

```@repl tuto
x = [1.0, 2.0, 3.0]
```

To compute its gradient, we need to choose a "backend", i.e. an AD package that DifferentiationInterface.jl will call under the hood.
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is very efficient for low-dimensional inputs, so we'll go with that one.
Backend types are defined and exported by [ADTypes.jl](https://github.com/SciML/ADTypes.jl):

```@repl tuto
backend = AutoForwardDiff()
```

Now we can use DifferentiationInterface.jl to get our gradient:

```@repl tuto
gradient(f, backend, x)
```

Was that fast?
We can use [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) to answer that question.

```@repl tuto
@btime gradient($f, $backend, $x);
```

More or less what you would get if you just used the API from ForwardDiff.jl:

```@repl tuto
@btime ForwardDiff.gradient($f, $x);
```

Not bad, but we can do better.

## Overwriting a gradient

Since we know how much space our gradient will occupy, we can pre-allocate that memory and offer it to AD.
Some backends can get a speed boost from this trick.

```@repl tuto
grad = zero(x)
grad = gradient!!(f, grad, backend, x)
```

Note the double exclamation mark, which is a convention telling you that `grad` _may or may not_ be overwritten, but will be returned either way (see [this section](@ref Variants) for more details).

```@repl tuto
@btime gradient!!($f, _grad, $backend, $x) evals=1 setup=(_grad=similar($x));
```

For some reason the in-place version is slower than our first attempt, but as you can see it has one less allocation, corresponding to the gradient vector.
Don't worry, we're not done yet.

## Preparing for multiple gradients

Internally, ForwardDiff.jl creates some data structures to keep track of things.
These objects can be reused between gradient computations, even on different input values.
We abstract away the preparation step behind a backend-agnostic syntax:

```@repl tuto
extras = prepare_gradient(f, backend, x)
```

You don't need to know what that is, you just need to pass it to the gradient operator.

```@repl tuto
grad = zero(x);
grad = gradient!!(f, grad, backend, x, extras)
```

Why, you ask?
Because it is much faster, and allocation-free.

```@repl tuto
@btime gradient!!($f, _grad, $backend, $x, _extras) evals=1 setup=(
    _grad=similar($x);
    _extras=prepare_gradient($f, $backend, $x)
);
```

## Switching backends

Now the whole point of DifferentiationInterface.jl is that you can easily experiment with different AD solutions.
Typically, for gradients, reverse mode AD might be a better fit.
So let's try the state-of-the-art [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)!

For this one, the backend definition is slightly more involved, because you need to feed the "mode" to the object from ADTypes.jl:

```@repl tuto
backend2 = AutoEnzyme(Enzyme.Reverse)
```

But once it is done, things run smoothly with exactly the same syntax:

```@repl tuto
gradient(f, backend2, x)
```

And we can run the same benchmarks:

```@repl tuto
@btime gradient!!($f, _grad, $backend2, $x, _extras) evals=1 setup=(
    _grad=similar($x);
    _extras=prepare_gradient($f, $backend2, $x)
);
```

Have you seen this?
It's blazingly fast.
And you know what's even better?
You didn't need to look at the docs of either ForwardDiff.jl or Enzyme.jl to achieve top performance with both, or to compare them.

## Testing and benchmarking

DifferentiationInterface.jl also provides some utilities for more involved comparison between backends.
They are gathered in a submodule.

```@repl tuto
using DifferentiationInterface.DifferentiationTest
```

The main entry point is [`test_differentiation`](@ref), which is used as follows:

```@repl tuto
data = test_differentiation(
    [AutoForwardDiff(), AutoEnzyme(Enzyme.Reverse)],  # backends to compare
    [gradient],  # operators to try
    [Scenario(f; x=x)];  # test scenario
    correctness=AutoZygote(),  # compare results to a "ground truth" from Zygote
    benchmark=true,  # measure runtime and allocations too
    detailed=true,  # print detailed test set
);
```

The output of `test_differentiation` when `benchmark=true` can be converted to a `DataFrame` from [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl):

```@repl tuto
df = DataFrames.DataFrame(pairs(data)...)
```

Here's what the resulting `DataFrame` looks like with all its columns.
Note that the results may be slightly different from the ones presented above (we use [Chairmarks.jl](https://github.com/LilithHafner/Chairmarks.jl) internally instead of BenchmarkTools.jl, and measure slightly different operators).

```@example tuto
import Markdown, PrettyTables  # hide
Markdown.parse(PrettyTables.pretty_table(String, df; backend=Val(:markdown), header=names(df)))  # hide
```
