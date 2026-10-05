"""
SciML form of an `EquationSystem`: loaded when SciMLBase is, so that OrdinaryDiffEq and
NonlinearSolve can drive Runes2P systems without making SciML a dependency of Runes2P.
"""
module Runes2PSciMLBaseExt

using Runes2P: Runes2P, EquationSystem, rhs!, mass_matrix, jacobian!, jacobian_prototype, jacobian_operator, lags_in_ode_form
using LinearAlgebra: mul!
using SciMLBase: ODEFunction

"""
    ode_function(system::EquationSystem)

`SciMLBase.ODEFunction` of `M du/dt = rhs(u, t)`: in-place right-hand side
`f(du, u, p, t)`, the mass matrix, and the Jacobian-vector product `jvp(Jv, v, u, p, t)`
(stored sparse Jacobian plus matrix-free terms). The sparse Jacobian itself (`jac`,
`jac_prototype`) is included only when it is the whole Jacobian, i.e. when the system has
no matrix-free and no explicit terms. In the ODE form explicit terms are part of `f` but
not of the Jacobian, which then only approximates `∂f/∂u` (as DREAM's explicit knock-on
term does). A system with a term that lags part of its residual to the start of the step
([`lags_in_ode_form`](@ref)) gets neither `jac` nor `jvp`: in the ODE form the start of the
step is the state itself, and the stored Jacobian leaves out the derivative of the lagged
part. The solver then takes its own derivatives. The parameter `p` is ignored.

Cost: `jvp` assembles the whole stored Jacobian at `u` on every call (one AD pass per
color, about 23 operator applications for the Fokker-Planck stencil) before the product; a
forward-mode pushforward seeded with `v` would cost one. Fine for occasional products; a
Krylov solver driven through SciML should reuse an assembled Jacobian instead.
"""
function Runes2P.ode_function(system::EquationSystem)
    f!(du, u, p, t) = (rhs!(du, u, system, t); nothing)
    cached = jacobian_prototype(system)
    function jvp!(Jv, v, u, p, t)
        jacobian!(cached, system, u, t)
        mul!(Jv, jacobian_operator(cached, system, u, t), v)
        return nothing
    end
    M = mass_matrix(system)
    # a term lagged to the start of the step: in the ODE form that is the state itself, and
    # the stored Jacobian leaves out the derivative of the lagged part
    any(lags_in_ode_form(term) for equation in system.equations for term in equation.terms) &&
        return ODEFunction{true}(f!; mass_matrix=M)
    if isempty(system.matrix_free_terms) && isempty(system.explicit_terms)
        jac!(J, u, p, t) = (jacobian!(J, system, u, t); nothing)
        return ODEFunction{true}(f!; mass_matrix=M, jac=jac!, jac_prototype=jacobian_prototype(system),
                                 jvp=jvp!)
    end
    return ODEFunction{true}(f!; mass_matrix=M, jvp=jvp!)
end

end # module Runes2PSciMLBaseExt
