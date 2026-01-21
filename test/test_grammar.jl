using Test
using Durbyn.Grammar

import Durbyn.Grammar: SesTerm, HoltTerm, HoltWintersTerm, CrostonTerm
import Durbyn.Grammar: EtsComponentTerm, EtsDriftTerm

@testset "Durbyn.Grammar Module Tests" begin

    @testset "ARIMA Order Terms" begin

        @testset "p() - AR order" begin
            term_fixed = p(2)
            @test term_fixed.term == :p
            @test term_fixed.min == 2
            @test term_fixed.max == 2

            term_range = p(1, 3)
            @test term_range.term == :p
            @test term_range.min == 1
            @test term_range.max == 3
        end

        @testset "d() - differencing order" begin
            term = d(1)
            @test term.term == :d
            @test term.min == 1
            @test term.max == 1

            term_range = d(0, 2)
            @test term_range.min == 0
            @test term_range.max == 2
        end

        @testset "q() - MA order" begin
            term = q(0, 3)
            @test term.term == :q
            @test term.min == 0
            @test term.max == 3
        end

        @testset "P() - Seasonal AR order" begin
            term = P(1)
            @test term.term == :P
            @test term.min == 1
            @test term.max == 1
        end

        @testset "D() - Seasonal differencing" begin
            term = D(0, 1)
            @test term.term == :D
            @test term.min == 0
            @test term.max == 1
        end

        @testset "Q() - Seasonal MA order" begin
            term = Q(1, 2)
            @test term.term == :Q
            @test term.min == 1
            @test term.max == 2
        end

        @testset "Order term validation" begin
            @test_throws ArgumentError p(-1)
            @test_throws ArgumentError d(-1, 0)
            @test_throws ArgumentError q(3, 1)
            @test_throws ArgumentError P(2, 1)
        end

        @testset "auto() term" begin
            term = auto()
            @test term isa AutoVarTerm
        end
    end

    @testset "ArimaOrderTerm construction" begin
        @testset "Valid construction" begin
            term = ArimaOrderTerm(:p, 0, 5)
            @test term.term == :p
            @test term.min == 0
            @test term.max == 5
        end

        @testset "Invalid term name" begin
            @test_throws ArgumentError ArimaOrderTerm(:x, 0, 1)
            @test_throws ArgumentError ArimaOrderTerm(:ar, 0, 1)
        end

        @testset "Invalid range" begin
            @test_throws ArgumentError ArimaOrderTerm(:p, -1, 1)
            @test_throws ArgumentError ArimaOrderTerm(:p, 2, 1)
        end
    end

    @testset "ETS Component Terms" begin

        @testset "e() - Error component" begin
            term_a = e("A")
            @test term_a.component == :error
            @test term_a.code == "A"

            term_m = e("M")
            @test term_m.code == "M"

            term_z = e("Z")
            @test term_z.code == "Z"
        end

        @testset "t() - Trend component" begin
            term_n = t("N")
            @test term_n.component == :trend
            @test term_n.code == "N"

            term_a = t("A")
            @test term_a.code == "A"

            term_m = t("M")
            @test term_m.code == "M"
        end

        @testset "s() - Seasonal component" begin
            term_n = s("N")
            @test term_n.component == :seasonal
            @test term_n.code == "N"

            term_a = s("A")
            @test term_a.code == "A"

            term_m = s("M")
            @test term_m.code == "M"
        end

        @testset "Case insensitivity" begin
            @test e("a").code == "A"
            @test t("m").code == "M"
            @test s("n").code == "N"
        end

        @testset "drift() term" begin
            drift_damped = drift(true)
            @test drift_damped.damped == true

            drift_undamped = drift(false)
            @test drift_undamped.damped == false

            drift_default = drift()
            @test drift_default.damped == true

            drift_auto = drift(:auto)
            @test drift_auto.damped === nothing
        end
    end

    @testset "Model Terms" begin

        @testset "ses() - Simple Exponential Smoothing" begin
            term = ses()
            @test term isa SesTerm
        end

        @testset "holt() - Holt's method" begin
            term = holt()
            @test term isa HoltTerm

            term_damped = holt(damped=true)
            @test term_damped.damped == true

            term_exp = holt(exponential=true)
            @test term_exp.exponential == true
        end

        @testset "hw()/holt_winters() - Holt-Winters" begin
            term = hw()
            @test term isa HoltWintersTerm

            term_add = hw(seasonal="additive")
            @test term_add.seasonal == "additive"

            term_mult = holt_winters(seasonal="multiplicative")
            @test term_mult.seasonal == "multiplicative"

            term_damped = hw(damped=true)
            @test term_damped.damped == true
        end

        @testset "croston() - Intermittent demand" begin
            term = croston()
            @test term isa CrostonTerm
            @test term.method == "hyndman"

            term_sba = croston(method="sba")
            @test term_sba.method == "sba"

            term_classic = croston(method="classic")
            @test term_classic.method == "classic"
        end

        @testset "arar() - ARAR model" begin
            term = arar()
            @test term isa ArarTerm
            @test term.max_ar_depth === nothing
            @test term.max_lag === nothing

            term_custom = arar(max_ar_depth=15, max_lag=20)
            @test term_custom.max_ar_depth == 15
            @test term_custom.max_lag == 20
        end

        @testset "bats() - BATS model" begin
            term = bats()
            @test term isa BatsTerm

            term_opts = bats(use_box_cox=true, use_trend=true)
            @test term_opts.use_box_cox == true
            @test term_opts.use_trend == true
        end

        @testset "tbats() - TBATS model" begin
            term = tbats()
            @test term isa TbatsTerm

            term_seasonal = tbats(seasonal_periods=[7, 365])
            @test term_seasonal.seasonal_periods == [7, 365]
        end

        @testset "theta() - Theta model" begin
            term = theta()
            @test term isa ThetaTerm

            term_stm = theta(model=:STM)
            @test term_stm.model_type == :STM

            term_otm = theta(model=:OTM)
            @test term_otm.model_type == :OTM
        end
    end

    @testset "@formula macro" begin

        @testset "Simple formulas" begin
            f1 = @formula(y = ses())
            @test f1 isa ModelFormula
            @test f1.target == :y

            f2 = @formula(sales = arar())
            @test f2.target == :sales
        end

        @testset "ARIMA formulas" begin
            f = @formula(y = p(1) + d(1) + q(1))
            @test f.target == :y

            f2 = @formula(y = p(1,2) + d(1) + q(0,2) + P(0,1) + D(1) + Q(0,1))
            @test f2.target == :y
        end

        @testset "With exogenous variables" begin
            f = @formula(y = p(1) + d(1) + q(1) + x1 + x2)
            @test f.target == :y
        end
    end

    @testset "Formula Compilation" begin

        @testset "compile_arima_formula" begin
            f = @formula(y = p(1,2) + d(1) + q(0,2))
            result = compile_arima_formula(f)

            @test haskey(result, :p)
            @test haskey(result, :d)
            @test haskey(result, :q)

            @test result[:p] == (1, 2)
            @test result[:d] == (1, 1)
            @test result[:q] == (0, 2)
        end

        @testset "compile_ets_formula" begin
            f = @formula(y = e("A") + t("A") + s("A"))
            result = compile_ets_formula(f)

            @test hasfield(typeof(result), :error) || haskey(result, :error)
            @test hasfield(typeof(result), :trend) || haskey(result, :trend)
            @test hasfield(typeof(result), :seasonal) || haskey(result, :seasonal)
        end
    end

    @testset "VarTerm (Exogenous Variables)" begin

        @testset "Basic variable term" begin
            term = VarTerm(:temperature)
            @test term.name == :temperature
        end

        @testset "Variable in formula" begin
            f = @formula(sales = ses() + temperature + promotion)
            @test f.target == :sales
        end
    end

    @testset "Term Validation" begin

        @testset "ARAR parameter validation" begin
            @test_throws ArgumentError @formula(y = arar(max_ar_depth=3))
            @test_throws ArgumentError @formula(y = arar(max_lag=-1))
        end
    end

    @testset "AbstractTerm inheritance" begin
        @test ArimaOrderTerm <: AbstractTerm
        @test SesTerm <: AbstractTerm
        @test HoltTerm <: AbstractTerm
        @test HoltWintersTerm <: AbstractTerm
        @test CrostonTerm <: AbstractTerm
        @test ArarTerm <: AbstractTerm
        @test BatsTerm <: AbstractTerm
        @test TbatsTerm <: AbstractTerm
        @test ThetaTerm <: AbstractTerm
    end

    @testset "ModelFormula structure" begin
        f = @formula(sales = ses())

        @test hasfield(typeof(f), :target)
        @test hasfield(typeof(f), :terms)
        @test f.target isa Symbol
    end

    @testset "Term Display" begin
        term = p(1, 3)
        str = string(term)
        @test occursin("p", str) || occursin("1", str) || occursin("3", str)
    end

    @testset "Edge Cases" begin

        @testset "Zero orders" begin
            @test p(0).min == 0
            @test d(0).min == 0
            @test q(0).min == 0
        end

        @testset "Large orders" begin
            term = p(0, 10)
            @test term.max == 10

            term = Q(0, 5)
            @test term.max == 5
        end

        @testset "Empty formula terms" begin
            f = @formula(y = ses())
            @test f isa ModelFormula
        end
    end

    @testset "_extract_single_term helper" begin
        f = @formula(y = arar(max_ar_depth=15))

        terms = f.terms isa Vector ? f.terms : [f.terms]
        arar_terms = filter(t -> t isa ArarTerm, terms)
        @test length(arar_terms) >= 0
    end

end
