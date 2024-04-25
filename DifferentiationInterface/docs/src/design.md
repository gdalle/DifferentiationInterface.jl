# Package design

## [Backend requirements](@id ssec-requirements)

To be usable with DifferentiationInterface.jl, an AD backend needs an object subtyping `ADTypes.AbstractADType`.
In addition, some operators must be defined:

| backend subtype        | pushforward necessary | pullback necessary |
| :--------------------- | :-------------------- | :----------------- |
| `ADTypes.ForwardMode`  | yes                   | no                 |
| `ADTypes.ReverseMode`  | no                    | yes                |
| `ADTypes.SymbolicMode` | yes                   | yes                |

Every backend we support corresponds to a package extension of DifferentiationInterface.jl (located in the `ext` subfolder).
Advanced users are welcome to code more backends and submit pull requests!
