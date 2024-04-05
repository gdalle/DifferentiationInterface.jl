```@meta
CurrentModule = Main
```

# Tutorial

We present a typical workflow with DifferentiationInterface.jl and showcase its potential performance benefits.

```@repl tuto
using DifferentiationInterface
import ForwardDiff, Enzyme
using BenchmarkTools
```

## Computing a gradient

A common use case of AD is optimizing real-valued functions with first- or second-order methods.
Let's define a simple objective

```@repl tuto
f(x::AbstractArray) = sum(abs2, x)
```

and a random input vector

```@repl tuto
x = [1.0, 2.0, 3.0];
```

To compute its gradient, we need to choose a "backend", i.e. an AD package that DifferentiationInterface.jl will call under the hood.
Most backend types are defined by [ADTypes.jl](https://github.com/SciML/ADTypes.jl) and re-exported by DifferentiationInterface.jl.
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is very generic and efficient for low-dimensional inputs, so it's a good starting point:

```@repl tuto
backend = AutoForwardDiff()
```

Now you can use DifferentiationInterface.jl to get the gradient:

```@repl tuto
gradient(f, backend, x)
```

Was that fast?
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) helps you answer that question.

```@repl tuto
@btime gradient($f, $backend, $x);
```

More or less what you would get if you just used the API from ForwardDiff.jl:

```@repl tuto
@btime ForwardDiff.gradient($f, $x);
```

Not bad, but you can do better.

## Overwriting a gradient

Since you know how much space your gradient will occupy, you can pre-allocate that memory and offer it to AD.
Some backends get a speed boost from this trick.

```@repl tuto
grad = zero(x)
grad = gradient!!(f, grad, backend, x)
```

Note the double exclamation mark, which is a convention telling you that `grad` _may or may not_ be overwritten, but will be returned either way (see [this section](@ref Variants) for more details).

```@repl tuto
@btime gradient!!($f, _grad, $backend, $x) evals=1 setup=(_grad=similar($x));
```

For some reason the in-place version is not much better than your first attempt.
However, it has one less allocation, which corresponds to the gradient vector you provided.
Don't worry, you're not done yet.

## Preparing for multiple gradients

Internally, ForwardDiff.jl creates some data structures to keep track of things.
These objects can be reused between gradient computations, even on different input values.
We abstract away the preparation step behind a backend-agnostic syntax:

```@repl tuto
extras = prepare_gradient(f, backend, x)
```

You don't need to know what this object is, you just need to pass it to the gradient operator.

```@repl tuto
grad = zero(x);
grad = gradient!!(f, grad, backend, x, extras)
```

Preparation makes the gradient computation much faster, and (in this case) allocation-free.

```@repl tuto
@btime gradient!!($f, _grad, $backend, $x, _extras) evals=1 setup=(
    _grad=similar($x);
    _extras=prepare_gradient($f, $backend, $x)
);
```

## Switching backends

The whole point of DifferentiationInterface.jl is that you can easily experiment with different AD solutions.
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

And you can run the same benchmarks:

```@repl tuto
@btime gradient!!($f, _grad, $backend2, $x, _extras) evals=1 setup=(
    _grad=similar($x);
    _extras=prepare_gradient($f, $backend2, $x)
);
```

Not only is it blazingly fast, you achieved this speedup without looking at the docs of either ForwardDiff.jl or Enzyme.jl!
In short, DifferentiationInterface.jl allows for easy testing and comparison of AD backends.
If you want to go further, check out the [DifferentiationTest.jl tutorial](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterfaceTest/dev/tutorial/).
