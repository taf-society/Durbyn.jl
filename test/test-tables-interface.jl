"""Unit tests for Tables.jl interface in holt_winters function."""

using Durbyn.ExponentialSmoothing: holt_winters, HoltWinters
using Durbyn: air_passengers

@testset "Tables.jl Interface Tests" begin

    @testset "holt_winters Tables.jl NamedTuple" begin
        ap = air_passengers()
        table_nt = (ap = ap,)
        model_nt = holt_winters(table_nt, 12)

        # test that it returns a HoltWinters object
        @test model_nt isa Durbyn.ExponentialSmoothing.HoltWinters

        # test that it produces the same result as the array version
        model_array = holt_winters(ap, 12)
        @test model_nt.fitted == model_array.fitted
        @test model_nt.residuals == model_array.residuals
        @test model_nt.aic == model_array.aic
    end

    @testset "holt_winters Tables.jl DataFrame" begin
        ap = air_passengers()
        df = DataFrame(ap = ap)
        model_df = holt_winters(df, 12)

        # test that it returns a HoltWinters object
        @test model_df isa Durbyn.ExponentialSmoothing.HoltWinters

        # test that it produces the same result as the array version
        model_array = holt_winters(ap, 12)
        @test model_df.fitted == model_array.fitted
        @test model_df.residuals == model_array.residuals
        @test model_df.aic == model_array.aic

        # specify column name
        model_df_col = holt_winters(df, 12; col = "ap")
        @test model_df_col.fitted == model_array.fitted
        @test model_df_col.residuals == model_array.residuals
        @test model_df_col.aic == model_array.aic
    end

end