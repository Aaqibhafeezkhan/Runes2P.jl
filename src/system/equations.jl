# Equations: the unknown of a row, the kind of the row and its terms. Moved out of
# equation_system.jl.

"""
    TimeDerivative()
    TimeDerivative(of)

Kind of [`Equation`](@ref) `du/dt = Σ terms`. The other kinds are [`Algebraic`](@ref) and
[`Constraint`](@ref). `TimeDerivative(of)` puts the time derivative of another unknown into
the row, `d(of)/dt = Σ terms`, for an unknown `of` of the same kind and sizes as the
unknown of the equation. Faraday's law is of this form: the equation of
the electric field is `dψ/dt = 2π E`, with the poloidal flux as `of`, and the equation of
the flux is Ampère's law, which is algebraic. Every algebraic row then holds its own
unknown. The unknown of an equation with the derivative of another unknown must appear in
the terms of that equation, or the equation of `of` must reach it through algebraic rows:
the temperature in DREAM's pairing of the energy of the cold electrons.
"""
struct TimeDerivative
    of::Union{Nothing,Symbol}
end
TimeDerivative() = TimeDerivative(nothing)

"""
    Algebraic()

Kind of [`Equation`](@ref) `0 = Σ terms` that holds its unknown. See also
[`TimeDerivative`](@ref) and [`Constraint`](@ref).
"""
struct Algebraic end

"""
    Constraint()

Kind of [`Equation`](@ref) `0 = Σ terms` that does not hold its unknown: the unknown is
fixed through the equations of other unknowns, which hold it. The magnitude of the particle
source is of this kind: its equation is the density that the distribution has to keep,
`0 = n(f) − n_cold − n_hot`, and the magnitude stands in the kinetic equation (DREAM's
`S_particle`). The system then has the index 2 in that unknown, which backward Euler
solves, and the diagonal entry of the row in the Jacobian is zero. A constraint names
the scale of the quantity it balances (`Equation(...; scale)`): its unknown is a rate, whose
scale does not fit the row.
"""
struct Constraint end

"""
    Equation(unknown, kind, terms...; scale=nothing)

Equation of the unknown `unknown` (a name in the layout): `TimeDerivative()`,
`TimeDerivative(of)` or `Algebraic()`, with the right-hand side the sum of `terms`
([`EquationTerm`](@ref)s).

`scale` is the magnitude of the quantity whose rate or balance the row holds, for the
weights of the residual ([`residual_scales`](@ref)). By default it is the scale of the
unknown whose time derivative the row holds, and the scale of the unknown of the equation
for an algebraic row. An algebraic row that balances another quantity than its unknown
names the scale of that quantity: the row of the poloidal flux holds Ampère's law.

`initial_value = :from_equation` makes [`integrate`](@ref) set the unknown of an algebraic
equation from its row at the start of a run (DREAM's `INITRULE_EVAL_EQUATION`), in place
of the value in the initial state: [`fill_initial_values!`](@ref). The default `:given`
takes the initial state as it is.
"""
struct Equation{K,TT<:Tuple}
    unknown::Symbol
    kind::K
    terms::TT
    scale::Union{Nothing,Float64}
    initial_value::Symbol
end
function Equation(unknown::Symbol, kind, terms::Tuple; scale::Union{Nothing,Real}=nothing,
                  initial_value::Symbol=:given)
    scale === nothing || scale > 0 || throw(ArgumentError("the scale of an equation must be positive"))
    initial_value in (:given, :from_equation) ||
        throw(ArgumentError("initial_value must be :given or :from_equation, not $initial_value"))
    initial_value === :from_equation && !(kind isa Algebraic) &&
        throw(ArgumentError("the equation of $unknown takes its initial value from its row, which needs an " *
                            "algebraic equation"))
    return Equation(unknown, kind, terms, scale === nothing ? nothing : Float64(scale), initial_value)
end
Equation(unknown::Symbol, kind, terms::EquationTerm...; scale::Union{Nothing,Real}=nothing,
         initial_value::Symbol=:given) =
    Equation(unknown, kind, terms; scale, initial_value)
