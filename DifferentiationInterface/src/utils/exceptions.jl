struct MissingBackendError <: Exception
    backend::AbstractADType
end
function Base.showerror(io::IO, e::MissingBackendError)
    println(io, "failed to use $(backend_str(e.backend)) backend.")
    if !check_available(e.backend)
        print(
            io,
            """Backend package is not loaded. To fix, run

              using $(backend_package_name(e.backend))
            """,
        )
    else
        print(
            io,
            "Please open an issue: https://github.com/gdalle/DifferentiationInterface.jl/issues/new",
        )
    end
end
