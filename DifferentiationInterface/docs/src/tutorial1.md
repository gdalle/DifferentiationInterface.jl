# Basics

We present the main features of DifferentiationInterface.jl.

```@example tuto1
using DifferentiationInterface
```

## Computing a gradient

A common use case of automatic differentiation (AD) is optimizing real-valued functions with first- or second-order methods.
Let's define a simple objective and a random input vector

```@example tuto1
f(x) = sum(abs2, x)

x = collect(1.0:5.0)
```

To compute its gradient, we need to choose a "backend", i.e. an AD package to call under the hood.
Most backend types are defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl) and re-exported by DifferentiationInterface.jl.

[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is very generic and efficient for low-dimensional inputs, so it's a good starting point:

```@example tuto1
import ForwardDiff

backend = AutoForwardDiff()
```

!!! tip
    To avoid name conflicts, load AD packages with `import` instead of `using`.
    Indeed, most AD packages also export operators like `gradient` and `jacobian`, but you only want to use the ones from DifferentiationInterface.jl.

Now you can use the following syntax to compute the gradient:

```@example tuto1
gradient(f, backend, x)
```

Was that fast?
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) helps you answer that question.

```@example tuto1
using BenchmarkTools

@benchmark gradient($f, $backend, $x)
```

Not bad, but you can do better.

## Overwriting a gradient

Since you know how much space your gradient will occupy (the same as your input `x`), you can pre-allocate that memory and offer it to AD.
Some backends get a speed boost from this trick.

```@example tuto1
grad = similar(x)
gradient!(f, grad, backend, x)
grad  # has been mutated
```

The bang indicates that one of the arguments of `gradient!` might be mutated.
More precisely, our convention is that _every positional argument between the function and the backend is mutated (and the `extras` too, see below)_.

```@example tuto1
@benchmark gradient!($f, _grad, $backend, $x) evals=1 setup=(_grad=similar($x))
```

For some reason the in-place version is not much better than your first attempt.
However, it makes fewer allocations, thanks to the gradient vector you provided.
Don't worry, you can get even more performance.

## Preparing for multiple gradients

Internally, ForwardDiff.jl creates some data structures to keep track of things.
These objects can be reused between gradient computations, even on different input values.
We abstract away the preparation step behind a backend-agnostic syntax:

```@example tuto1
extras = prepare_gradient(f, backend, randn(eltype(x), size(x)))
```

You don't need to know what this object is, you just need to pass it to the gradient operator.
Note that preparation does not depend on the actual components of the vector `x`, just on its type and size.
You can thus reuse the `extras` for different values of the input.

```@example tuto1
grad = similar(x)
gradient!(f, grad, backend, x, extras)
grad  # has been mutated
```

Preparation makes the gradient computation much faster, and (in this case) allocation-free.

```@example tuto1
@benchmark gradient!($f, _grad, $backend, $x, _extras) evals=1 setup=(
    _grad=similar($x);
    _extras=prepare_gradient($f, $backend, $x)
)
```

Beware that the `extras` object is nearly always mutated by differentiation operators, even though it is given as the last positional argument.

## Switching backends

The whole point of DifferentiationInterface.jl is that you can easily experiment with different AD solutions.
Typically, for gradients, reverse mode AD might be a better fit, so let's try the state-of-the-art [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)!

```@example tuto1
import Enzyme

backend2 = AutoEnzyme()
```

Once the backend is created, things run smoothly with exactly the same syntax as before:

```@example tuto1
gradient(f, backend2, x)
```

And you can run the same benchmarks to see what you gained (although such a small input may not be realistic):

```@example tuto1
@benchmark gradient!($f, _grad, $backend2, $x, _extras) evals=1 setup=(
    _grad=similar($x);
    _extras=prepare_gradient($f, $backend2, $x)
)
```

In short, DifferentiationInterface.jl allows for easy testing and comparison of AD backends.
If you want to go further, check out the [documentation of DifferentiationInterfaceTest.jl](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterfaceTest).
This related package provides benchmarking utilities to compare backends and help you select the one that is best suited for your problem.
