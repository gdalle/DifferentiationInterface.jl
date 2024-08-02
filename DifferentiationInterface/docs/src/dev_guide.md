# Dev guide

This page is important reading if you want to contribute to DifferentiationInterface.jl.
It is not part of the public API.

!!! warning
    The content below may become outdated, in which case you should refer to the source code as the ground truth.

## General principles

The package is structured around 8 [operators](@ref Operators):

- [`derivative`](@ref)
- [`second_derivative`](@ref)
- [`gradient`](@ref)
- [`jacobian`](@ref)
- [`hessian`](@ref)
- [`pushforward`](@ref)
- [`pullback`](@ref)
- [`hvp`](@ref)

Most operators have 4 variants, which look like this in the first order: `operator`, `operator!`, `value_and_operator`, `value_and_operator!`.

## New operator

To implement a new operator for an existing backend, you need to write 5 methods: 1 for [preparation](@ref Preparation) and 4 corresponding to the variants of the operator (see above).
In some cases, a subset of those methods will be enough, but most of the time, forgetting one will trigger errors.
For first-order operators, you may also want to support [two-argument functions](@ref "Mutation and signatures"), which requires another 5 methods (defined on `f!` instead of `f`).

The method `prepare_operator` must output an `extras` object of the correct type.
For instance, `prepare_gradient(f, backend, x)` must return a [`DifferentiationInterface.GradientExtras`](@ref).
Assuming you don't need any preparation for said operator, you can use the trivial extras that are already defined, like `DifferentiationInterface.NoGradientExtras`.
Otherwise, define a custom struct like `MyGradientExtras <: DifferentiationInterface.GradientExtras` and put the necessary storage in there.

## New backend

If you want to implement a new backend, for instance because you developed a new AD package called `SuperDiff`, please open a pull request to DifferentiationInterface.jl.
Your AD package needs to be registered first.

### Core code

In the main package, you should define a new struct `SuperDiffBackend` which subtypes [`ADTypes.AbstractADType`](@extref ADTypes), and endow it with the fields you need to parametrize your differentiation routines.
You also have to define [`ADTypes.mode`](@extref) and [`DifferentiationInterface.twoarg_support`](@ref) on `SuperDiffBackend`.

!!! info
    In the end, this backend struct will need to be contributed to [ADTypes.jl](https://github.com/SciML/ADTypes.jl).
    However, putting it in the DifferentiationInterface.jl PR is a good first step for debugging.

In a [package extension](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)) named `DifferentiationInterfaceSuperDiffExt`, you need to implement at least [`pushforward`](@ref) or [`pullback`](@ref) (and their variants).
The exact requirements depend on the differentiation mode you chose:

| backend mode                                      | pushforward necessary | pullback necessary |
| :------------------------------------------------ | :-------------------- | :----------------- |
| [`ADTypes.ForwardMode`](@extref ADTypes)          | yes                   | no                 |
| [`ADTypes.ReverseMode`](@extref ADTypes)          | no                    | yes                |
| [`ADTypes.ForwardOrReverseMode`](@extref ADTypes) | yes                   | yes                |
| [`ADTypes.SymbolicMode`](@extref ADTypes)         | yes                   | yes                |

Every other operator can be deduced from these two, but you can gain efficiency by implementing additional operators.

### Tests and docs

Once that is done, you need to add your new backend to the test suite.
Test files should be gathered in a folder named `SuperDiff` inside [`DifferentiationInterface/test/Back`](https://github.com/gdalle/DifferentiationInterface.jl/tree/main/DifferentiationInterface/test/Back).
They should use [DifferentiationInterfaceTest.jl](https://github.com/gdalle/DifferentiationInterface.jl/tree/main/DifferentiationInterfaceTest) to check correctness against the default scenarios.
Take inspiration from the tests of other backends to write your own.
To activate tests in CI, modify the [test workflow](https://github.com/gdalle/DifferentiationInterface.jl/blob/main/.github/workflows/Test.yml) and add your package to the list.
To run the tests locally, replace the following line in [`DifferentiationInterface/test/runtests.jl`](https://github.com/gdalle/DifferentiationInterface.jl/blob/main/DifferentiationInterface/test/runtests.jl)

```julia
GROUP = get(ENV, "JULIA_DI_TEST_GROUP", "All")
```

with the much cheaper version

```julia
GROUP = get(ENV, "JULIA_DI_TEST_GROUP", "Back/SuperDiff")
```

but don't forget to switch it back before pushing.

Finally, you need to add your backend to the documentation, modifying every page that involves a list of backends.
That includes the README.
