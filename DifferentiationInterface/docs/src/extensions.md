# Package extensions

```@meta
CurrentModule = DifferentiationInterface
```

Package extension content is not part of the public API.
If any docstrings are present in an extension, they will appear below.

```@autodocs
Modules = [
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceChainRulesCoreExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceDiffractorExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceEnzymeExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceFastDifferentiationExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceFiniteDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceFiniteDifferencesExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceForwardDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfacePolyesterForwardDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceReverseDiffExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceSymbolicsExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceTapirExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceTrackerExt),
    Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceZygoteExt)
]
Filter = t -> !(t isa Type && t <: ADTypes.AbstractADType)
```
