var documenterSearchIndex = {"docs":
[{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"CurrentModule = Main\nCollapsedDocStrings = true","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"DifferentiationInterfaceTest","category":"page"},{"location":"api/#DifferentiationInterfaceTest","page":"API reference","title":"DifferentiationInterfaceTest","text":"DifferentiationInterfaceTest\n\nTesting and benchmarking utilities for automatic differentiation in Julia.\n\nExports\n\nDifferentiationBenchmarkDataRow\nFIRST_ORDER\nSECOND_ORDER\nScenario\nbenchmark_differentiation\ncomponent_scenarios\ndefault_scenarios\ngpu_scenarios\nsparse_scenarios\nstatic_scenarios\ntest_differentiation\n\n\n\n\n\n","category":"module"},{"location":"api/#Entry-points","page":"API reference","title":"Entry points","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Scenario\ntest_differentiation\nbenchmark_differentiation\nFIRST_ORDER\nSECOND_ORDER","category":"page"},{"location":"api/#DifferentiationInterfaceTest.Scenario","page":"API reference","title":"DifferentiationInterfaceTest.Scenario","text":"Scenario{op,pl_op,pl_fun}\n\nStore a testing scenario composed of a function and its input + output for a given operator.\n\nThis generic type should never be used directly: use the specific constructor corresponding to the operator you want to test, or a predefined list of scenarios.\n\nType parameters\n\nop: one  of :pushforward, :pullback, :derivative, :gradient, :jacobian,:second_derivative, :hvp, :hessian\npl_op: either :in (for op!(f, result, backend, x)) or :out (for result = op(f, backend, x))\npl_fun: either :in (for f!(y, x)) or :out (for y = f(x))\n\nConstructors\n\nScenario{op,pl_op}(f, x; tang, contexts, res1, res2)\nScenario{op,pl_op}(f!, y, x; tang, contexts, res1, res2)\n\nFields\n\nf::Any: function f (if args==1) or f! (if args==2) to apply\nx::Any: primal input\ny::Any: primal output\ntang::Union{Nothing, NTuple{N, T} where {N, T}}: tangents for pushforward, pullback or HVP\ncontexts::Tuple: contexts (if applicable)\nres1::Any: first-order result of the operator (if applicable)\nres2::Any: second-order result of the operator (if applicable)\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.test_differentiation","page":"API reference","title":"DifferentiationInterfaceTest.test_differentiation","text":"test_differentiation(\n    backends::Vector{<:ADTypes.AbstractADType};\n    ...\n) -> Union{Nothing, DataFrames.DataFrame}\ntest_differentiation(\n    backends::Vector{<:ADTypes.AbstractADType},\n    scenarios::Vector{<:Scenario};\n    correctness,\n    type_stability,\n    benchmark,\n    excluded,\n    detailed,\n    logging,\n    isapprox,\n    atol,\n    rtol,\n    scenario_intact,\n    sparsity,\n    ignored_modules,\n    function_filter,\n    count_calls,\n    benchmark_test\n) -> Union{Nothing, DataFrames.DataFrame}\n\n\nApply a list of backends on a list of scenarios, running a variety of different tests and/or benchmarks.\n\nReturn\n\nThis function always creates and runs a @testset, though its contents may vary.\n\nif benchmark == :none, it returns nothing.\nif benchmark != :none, it returns a DataFrame of benchmark results, whose columns correspond to the fields of DifferentiationBenchmarkDataRow.\n\nPositional arguments\n\nbackends::Vector{<:AbstractADType}: the backends to test\nscenarios::Vector{<:Scenario}: the scenarios on which to test them (defaults to the output of default_scenarios())\n\nKeyword arguments\n\nTest categories:\n\ncorrectness=true: whether to compare the differentiation results with the theoretical values specified in each scenario\ntype_stability=:none: whether (and how) to check type stability of operators with JET.jl.\nbenchmark=:none: whether (and how) to benchmark operators with Chairmarks.jl\n\nFor type_stability and benchmark, the possible values are :none, :prepared or :full, each concerns a different subset of calls:\n\nkwarg prepared operator unprepared operator preparation\n:none no no no\n:prepared yes no no\n:full yes yes yes\n\nMisc options:\n\nexcluded::Vector{Symbol}: list of operators to exclude, such as FIRST_ORDER or SECOND_ORDER\ndetailed=false: whether to create a detailed or condensed testset\nlogging=false: whether to log progress\n\nCorrectness options:\n\nisapprox=isapprox: function used to compare objects approximately, with the standard signature isapprox(x, y; atol, rtol)\natol=0: absolute precision for correctness testing (when comparing to the reference outputs)\nrtol=1e-3: relative precision for correctness testing (when comparing to the reference outputs)\nscenario_intact=true: whether to check that the scenario remains unchanged after the operators are applied\nsparsity=false: whether to check sparsity patterns for Jacobians / Hessians\n\nType stability options:\n\nignored_modules=nothing: list of modules that JET.jl should ignore\nfunction_filter: filter for functions that JET.jl should ignore (with a reasonable default)\n\nBenchmark options:\n\ncount_calls=true: whether to also count function calls during benchmarking\nbenchmark_test=true: whether to include tests which succeed iff benchmark doesn't error\n\n\n\n\n\ntest_differentiation(\n    backend::ADTypes.AbstractADType,\n    args...;\n    kwargs...\n) -> Union{Nothing, DataFrames.DataFrame}\n\n\nShortcut for a single backend.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.benchmark_differentiation","page":"API reference","title":"DifferentiationInterfaceTest.benchmark_differentiation","text":"benchmark_differentiation(\n    backends,\n    scenarios::Vector{<:Scenario};\n    benchmark,\n    excluded,\n    logging,\n    count_calls,\n    benchmark_test\n) -> Union{Nothing, DataFrames.DataFrame}\n\n\nShortcut for test_differentiation with only benchmarks and no correctness or type stability checks.\n\nSpecifying the set of scenarios is mandatory for this function.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.FIRST_ORDER","page":"API reference","title":"DifferentiationInterfaceTest.FIRST_ORDER","text":"FIRST_ORDER = [:pushforward, :pullback, :derivative, :gradient, :jacobian]\n\nList of all first-order operators, to facilitate exclusion during tests.\n\n\n\n\n\n","category":"constant"},{"location":"api/#DifferentiationInterfaceTest.SECOND_ORDER","page":"API reference","title":"DifferentiationInterfaceTest.SECOND_ORDER","text":"SECOND_ORDER = [:hvp, :second_derivative, :hessian]\n\nList of all second-order operators, to facilitate exclusion during tests.\n\n\n\n\n\n","category":"constant"},{"location":"api/#Utilities","page":"API reference","title":"Utilities","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"DifferentiationInterfaceTest.DifferentiationBenchmarkDataRow","category":"page"},{"location":"api/#DifferentiationInterfaceTest.DifferentiationBenchmarkDataRow","page":"API reference","title":"DifferentiationInterfaceTest.DifferentiationBenchmarkDataRow","text":"DifferentiationBenchmarkDataRow\n\nAd-hoc storage type for differentiation benchmarking results.\n\nFields\n\nbackend::ADTypes.AbstractADType: backend used for benchmarking\nscenario::Scenario: scenario used for benchmarking\noperator::Symbol: differentiation operator used for benchmarking, e.g. :gradient or :hessian\nprepared::Union{Nothing, Bool}: whether the operator had been prepared\ncalls::Int64: number of calls to the differentiated function for one call to the operator\nsamples::Int64: number of benchmarking samples taken\nevals::Int64: number of evaluations used for averaging in each sample\ntime::Float64: minimum runtime over all samples, in seconds\nallocs::Float64: minimum number of allocations over all samples\nbytes::Float64: minimum memory allocated over all samples, in bytes\ngc_fraction::Float64: minimum fraction of time spent in garbage collection over all samples, between 0.0 and 1.0\ncompile_fraction::Float64: minimum fraction of time spent compiling over all samples, between 0.0 and 1.0\n\nSee the documentation of Chairmarks.jl for more details on the measurement fields.\n\n\n\n\n\n","category":"type"},{"location":"api/#Pre-made-scenario-lists","page":"API reference","title":"Pre-made scenario lists","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"The precise contents of the scenario lists are not part of the API, only their existence.","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"default_scenarios\nsparse_scenarios\ncomponent_scenarios\ngpu_scenarios\nstatic_scenarios","category":"page"},{"location":"api/#DifferentiationInterfaceTest.default_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.default_scenarios","text":"default_scenarios(rng=Random.default_rng())\n\nCreate a vector of Scenarios with standard array types.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.sparse_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.sparse_scenarios","text":"sparse_scenarios(rng=Random.default_rng())\n\nCreate a vector of Scenarios with sparse array types, focused on sparse Jacobians and Hessians.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.component_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.component_scenarios","text":"component_scenarios(rng=Random.default_rng())\n\nCreate a vector of Scenarios with component array types from ComponentArrays.jl.\n\nwarning: Warning\nThis function requires ComponentArrays.jl to be loaded (it is implemented in a package extension).\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.gpu_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.gpu_scenarios","text":"gpu_scenarios(rng=Random.default_rng())\n\nCreate a vector of Scenarios with GPU array types from JLArrays.jl.\n\nwarning: Warning\nThis function requires JLArrays.jl to be loaded (it is implemented in a package extension).\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.static_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.static_scenarios","text":"static_scenarios(rng=Random.default_rng())\n\nCreate a vector of Scenarios with static array types from StaticArrays.jl.\n\nwarning: Warning\nThis function requires StaticArrays.jl to be loaded (it is implemented in a package extension).\n\n\n\n\n\n","category":"function"},{"location":"api/#Internals","page":"API reference","title":"Internals","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"This is not part of the public API.","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [DifferentiationInterfaceTest]\nPublic = false","category":"page"},{"location":"api/#Base.zero-Union{Tuple{Scenario{op, pl_op, pl_fun}}, Tuple{pl_fun}, Tuple{pl_op}, Tuple{op}} where {op, pl_op, pl_fun}","page":"API reference","title":"Base.zero","text":"zero(scen::Scenario)\n\nReturn a new Scenario identical to scen except for the first- and second-order results which are set to zero.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterfaceTest.allocfree_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.allocfree_scenarios","text":"allocfree_scenarios(rng::AbstractRNG=default_rng())\n\nCreate a vector of Scenarios with functions that do not allocate.\n\nwarning: Warning\nAt the moment, second-order scenarios are excluded.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.batchify-Union{Tuple{Scenario{op, pl_op, pl_fun}}, Tuple{pl_fun}, Tuple{pl_op}, Tuple{op}} where {op, pl_op, pl_fun}","page":"API reference","title":"DifferentiationInterfaceTest.batchify","text":"batchify(scen::Scenario)\n\nReturn a new Scenario identical to scen except for the tangents tang and associated results res1 / res2, which are duplicated (batch mode).\n\nOnly works if scen is a pushforward, pullback or hvp scenario.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterfaceTest.change_function-Union{Tuple{pl_fun}, Tuple{pl_op}, Tuple{op}, Tuple{Scenario{op, pl_op, pl_fun}, Any}} where {op, pl_op, pl_fun}","page":"API reference","title":"DifferentiationInterfaceTest.change_function","text":"change_function(scen::Scenario, new_f)\n\nReturn a new Scenario identical to scen except for the function f which is changed to new_f.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterfaceTest.closurify-Tuple{Scenario}","page":"API reference","title":"DifferentiationInterfaceTest.closurify","text":"closurify(scen::Scenario)\n\nReturn a new Scenario identical to scen except for the function f which is made to close over differentiable data.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterfaceTest.constantify-Union{Tuple{Scenario{op, pl_op, pl_fun}}, Tuple{pl_fun}, Tuple{pl_op}, Tuple{op}} where {op, pl_op, pl_fun}","page":"API reference","title":"DifferentiationInterfaceTest.constantify","text":"constantify(scen::Scenario)\n\nReturn a new Scenario identical to scen except for the function f, which is made to accept an additional constant argument a by which the output is multiplied. The output and result fields are updated accordingly.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterfaceTest.flux_isapprox","page":"API reference","title":"DifferentiationInterfaceTest.flux_isapprox","text":"flux_isapprox(x, y; atol, rtol)\n\nApproximate comparison function to use in correctness tests with gradients of Flux.jl networks.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.flux_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.flux_scenarios","text":"flux_scenarios(rng=Random.default_rng())\n\nCreate a vector of Scenarios with neural networks from Flux.jl.\n\nwarning: Warning\nThis function requires FiniteDifferences.jl and Flux.jl to be loaded (it is implemented in a package extension).\n\ndanger: Danger\nThese scenarios are still experimental and not part of the public API. Their ground truth values are computed with finite differences, and thus subject to imprecision.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.lux_isapprox","page":"API reference","title":"DifferentiationInterfaceTest.lux_isapprox","text":"lux_isapprox(x, y; atol, rtol)\n\nApproximate comparison function to use in correctness tests with gradients of Lux.jl networks.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.lux_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.lux_scenarios","text":"lux_scenarios(rng=Random.default_rng())\n\nCreate a vector of Scenarios with neural networks from Lux.jl.\n\nwarning: Warning\nThis function requires ComponentArrays.jl, ForwardDiff.jl, Lux.jl and LuxTestUtils.jl to be loaded (it is implemented in a package extension).\n\ndanger: Danger\nThese scenarios are still experimental and not part of the public API.\n\n\n\n\n\n","category":"function"},{"location":"#DifferentiationInterfaceTest","page":"Home","title":"DifferentiationInterfaceTest","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Build Status) (Image: Coverage) (Image: Code Style: Blue) (Image: ColPrac: Contributor's Guide on Collaborative Practices for Community Packages) (Image: DOI)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Package Docs\nDifferentiationInterface (Image: Stable)     (Image: Dev)\nDifferentiationInterfaceTest (Image: Stable) (Image: Dev)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Testing and benchmarking utilities for automatic differentiation (AD) in Julia, based on DifferentiationInterface.","category":"page"},{"location":"#Goal","page":"Home","title":"Goal","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Make it easy to know, for a given function:","category":"page"},{"location":"","page":"Home","title":"Home","text":"which AD backends can differentiate it\nhow fast they can do it","category":"page"},{"location":"#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Predefined or custom test scenarios\nCorrectness tests\nType stability tests\nCount calls to the function\nBenchmark runtime and allocations\nScenarios with weird array types (GPU, static) in package extensions","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install the stable version of the package, run the following code in a Julia REPL:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\n\nPkg.add(\"DifferentiationInterfaceTest\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"To install the development version, run this instead:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\n\nPkg.add(\n    url=\"https://github.com/gdalle/DifferentiationInterface.jl\",\n    subdir=\"DifferentiationInterface\"\n)\n    \nPkg.add(\n    url=\"https://github.com/gdalle/DifferentiationInterface.jl\",\n    subdir=\"DifferentiationInterfaceTest\"\n)","category":"page"},{"location":"tutorial/#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"We present a typical workflow with DifferentiationInterfaceTest.jl, building on the tutorial of the DifferentiationInterface.jl documentation (which we encourage you to read first).","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using DifferentiationInterface, DifferentiationInterfaceTest\nimport ForwardDiff, Zygote","category":"page"},{"location":"tutorial/#Introduction","page":"Tutorial","title":"Introduction","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"The AD backends we want to compare are ForwardDiff.jl and Enzyme.jl.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"backends = [AutoForwardDiff(), AutoZygote()]","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"To do that, we are going to take gradients of a simple function:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"f(x::AbstractArray) = sum(sin, x)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Of course we know the true gradient mapping:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"∇f(x::AbstractArray) = cos.(x)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"DifferentiationInterfaceTest.jl relies with so-called \"scenarios\", in which you encapsulate the information needed for your test:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"the operator category (:gradient)\nthe behavior of the operator (either :in or :out of place)\nthe function f\nthe input x of the function f\nthe reference first-order result res1 of the operator","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"xv = rand(Float32, 3)\nxm = rand(Float64, 3, 2)\nscenarios = [\n    Scenario{:gradient,:out}(f, xv; res1=∇f(xv)),\n    Scenario{:gradient,:out}(f, xm; res1=∇f(xm))\n];\nnothing  # hide","category":"page"},{"location":"tutorial/#Testing","page":"Tutorial","title":"Testing","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"The main entry point for testing is the function test_differentiation. It has many options, but the main ingredients are the following:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"test_differentiation(\n    backends,  # the backends you want to compare\n    scenarios,  # the scenarios you defined,\n    correctness=true,  # compares values against the reference\n    type_stability=:none,  # checks type stability with JET.jl\n    detailed=true,  # prints a detailed test set\n)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"If you are too lazy to manually specify the reference, you can also provide an AD backend as the ref_backend keyword argument, which will serve as the ground truth for comparison.","category":"page"},{"location":"tutorial/#Benchmarking","page":"Tutorial","title":"Benchmarking","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Once you are confident that your backends give the correct answers, you probably want to compare their performance. This is made easy by the benchmark_differentiation function, whose syntax should feel familiar:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"df = benchmark_differentiation(backends, scenarios);","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"The resulting object is a DataFrame from DataFrames.jl, whose columns correspond to the fields of DifferentiationBenchmarkDataRow:","category":"page"}]
}
