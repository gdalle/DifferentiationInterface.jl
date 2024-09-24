using ADTypes: ADTypes, AbstractADType
using DifferentiationInterface
import DifferentiationInterface as DI
using Test

"""
    AutoBrokenForward <: ADTypes.AbstractADType

Available forward-mode backend with no pushforward implementation.
Used to test error messages.
"""
struct AutoBrokenForward <: AbstractADType end
ADTypes.mode(::AutoBrokenForward) = ADTypes.ForwardMode()
DI.check_available(::AutoBrokenForward) = true

"""
    AutoBrokenReverse <: ADTypes.AbstractADType

Available reverse-mode backend with no pullback implementation.
Used to test error messages. 
"""
struct AutoBrokenReverse <: AbstractADType end
ADTypes.mode(::AutoBrokenReverse) = ADTypes.ReverseMode()

## Test exceptions
@testset "MissingBackendError" begin
    x = [1.0]
    y = similar(x)

    @test_throws DI.MissingBackendError jacobian(copy, AutoBrokenForward(), x)
    @test_throws DI.MissingBackendError jacobian(copy, AutoBrokenReverse(), x)

    @test_throws DI.MissingBackendError jacobian(copyto!, y, AutoBrokenForward(), x)
    @test_throws DI.MissingBackendError jacobian(copyto!, y, AutoBrokenReverse(), x)

    @test_throws DI.MissingBackendError hessian(sum, AutoBrokenForward(), x)
    @test_throws DI.MissingBackendError hessian(sum, AutoBrokenReverse(), x)

    sprint(showerror, DI.MissingBackendError(AutoBrokenForward()))
    sprint(showerror, DI.MissingBackendError(AutoBrokenReverse()))
end
