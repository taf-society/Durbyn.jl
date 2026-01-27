using Test
using Durbyn
using Dates

@testset "PanelData Refactoring Tests" begin

    @testset "Backward compatibility" begin
        data = (
            store = ["A", "A", "A", "B", "B", "B"],
            date = [Date(2024,1,1), Date(2024,1,2), Date(2024,1,3),
                    Date(2024,1,1), Date(2024,1,2), Date(2024,1,3)],
            sales = [100, 110, 120, 200, 210, 220]
        )

        panel = PanelData(data; groupby=:store, date=:date, m=12)

        @test panel.groups == [:store]
        @test panel.date == :date
        @test panel.m == 12
        @test panel.frequency === nothing
        @test panel.target === nothing
    end

    @testset "Frequency to m inference" begin
        data = (
            store = ["A", "A"],
            date = [Date(2024,1,1), Date(2024,1,2)],
            sales = [100, 110]
        )

        panel = PanelData(data; groupby=:store, date=:date, frequency=:daily)
        @test panel.m == 7
        @test panel.frequency == :daily

        panel2 = PanelData(data; groupby=:store, date=:date, frequency=:monthly)
        @test panel2.m == 12

        panel3 = PanelData(data; groupby=:store, date=:date, frequency=:hourly)
        @test panel3.m == 24
    end

    @testset "Multi-seasonality" begin
        data = (
            store = ["A", "A"],
            date = [Date(2024,1,1), Date(2024,1,2)],
            sales = [100, 110]
        )

        panel = PanelData(data; groupby=:store, date=:date, m=[7, 365])
        @test panel.m == [7, 365]
    end

    @testset "Fill time gaps" begin
        data_gaps = (
            store = ["A", "A", "A"],
            date = [Date(2024,1,1), Date(2024,1,2), Date(2024,1,5)],
            sales = [100, 110, 150]
        )

        panel = PanelData(data_gaps;
            groupby=:store,
            date=:date,
            frequency=:daily,
            fill_time=true
        )

        @test length(panel.data.date) == 5
        @test panel.time_fill_meta !== nothing
        @test panel.time_fill_meta.n_added == 2
    end

    @testset "Target gap filling - LOCF" begin
        data = (
            store = ["A", "A", "A", "A"],
            date = [Date(2024,1,1), Date(2024,1,2), Date(2024,1,3), Date(2024,1,4)],
            sales = [100, missing, missing, 140]
        )

        panel = PanelData(data;
            groupby=:store,
            date=:date,
            frequency=:daily,
            target=:sales,
            target_na=(strategy=:locf,)
        )

        @test panel.data.sales[2] == 100
        @test panel.data.sales[3] == 100
        @test panel.data.sales[4] == 140

        @test panel.data.sales_is_imputed[1] == false
        @test panel.data.sales_is_imputed[2] == true
        @test panel.data.sales_is_imputed[3] == true
        @test panel.data.sales_is_imputed[4] == false

        @test panel.target_meta !== nothing
        @test panel.target_meta.n_imputed == 2
    end

    @testset "Target gap filling - NOCB" begin
        data = (
            store = ["A", "A", "A", "A"],
            date = [Date(2024,1,1), Date(2024,1,2), Date(2024,1,3), Date(2024,1,4)],
            sales = [100, missing, missing, 140]
        )

        panel = PanelData(data;
            groupby=:store,
            date=:date,
            frequency=:daily,
            target=:sales,
            target_na=(strategy=:nocb,)
        )

        @test panel.data.sales[2] == 140
        @test panel.data.sales[3] == 140
    end

    @testset "Target gap filling - Linear" begin
        data = (
            store = ["A", "A", "A", "A", "A"],
            date = [Date(2024,1,1), Date(2024,1,2), Date(2024,1,3), Date(2024,1,4), Date(2024,1,5)],
            sales = [100.0, missing, missing, missing, 200.0]
        )

        panel = PanelData(data;
            groupby=:store,
            date=:date,
            frequency=:daily,
            target=:sales,
            target_na=(strategy=:linear,)
        )

        @test panel.data.sales[2] ≈ 125.0
        @test panel.data.sales[3] ≈ 150.0
        @test panel.data.sales[4] ≈ 175.0
    end

    @testset "Exogenous gap filling" begin
        data = (
            store = ["A", "A", "A", "A"],
            date = [Date(2024,1,1), Date(2024,1,2), Date(2024,1,3), Date(2024,1,4)],
            sales = [100, 110, 120, 130],
            price = [10.0, missing, missing, 13.0]
        )

        panel = PanelData(data;
            groupby=:store,
            date=:date,
            frequency=:daily,
            xreg_na=Dict(:price => (strategy=:locf,))
        )

        @test panel.data.price[2] == 10.0
        @test panel.data.price[3] == 10.0
        @test haskey(panel.xreg_meta, :price)
        @test panel.xreg_meta[:price].n_imputed == 2
    end

    @testset "Full preprocessing pipeline" begin
        data = (
            store = ["A", "A", "A", "B", "B"],
            date = [Date(2024,1,1), Date(2024,1,2), Date(2024,1,5),
                    Date(2024,1,1), Date(2024,1,4)],
            sales = [100, missing, 150, 200, missing],
            price = [10.0, 10.5, missing, 20.0, missing]
        )

        panel = PanelData(data;
            groupby=:store,
            date=:date,
            frequency=:daily,
            target=:sales,
            fill_time=true,
            target_na=(strategy=:locf,),
            xreg_na=Dict(:price => (strategy=:locf,))
        )

        @test panel.time_fill_meta !== nothing
        @test panel.target_meta !== nothing
        @test haskey(panel.xreg_meta, :price)

        @test panel.time_fill_meta.n_added > 0
    end

    @testset "Balanced panel" begin
        data = (
            store = ["A", "A", "A", "B", "B", "B", "B"],
            date = [Date(2024,1,3), Date(2024,1,4), Date(2024,1,5),
                    Date(2024,1,1), Date(2024,1,2), Date(2024,1,3), Date(2024,1,4)],
            sales = [130, 140, 150, 100, 110, 120, 130]
        )

        panel_unbalanced = PanelData(data;
            groupby=:store,
            date=:date,
            frequency=:daily,
            fill_time=true,
            balanced=false
        )

        panel_balanced = PanelData(data;
            groupby=:store,
            date=:date,
            frequency=:daily,
            fill_time=true,
            balanced=true
        )

        @test length(panel_unbalanced.data.date) == 7
        @test length(panel_balanced.data.date) == 10

        store_a_dates = [panel_balanced.data.date[i]
                        for i in 1:length(panel_balanced.data.date)
                        if panel_balanced.data.store[i] == "A"]
        @test length(store_a_dates) == 5
        @test minimum(store_a_dates) == Date(2024, 1, 1)
        @test maximum(store_a_dates) == Date(2024, 1, 5)

        store_b_dates = [panel_balanced.data.date[i]
                        for i in 1:length(panel_balanced.data.date)
                        if panel_balanced.data.store[i] == "B"]
        @test length(store_b_dates) == 5
        @test minimum(store_b_dates) == Date(2024, 1, 1)
        @test maximum(store_b_dates) == Date(2024, 1, 5)

        store_a_sales = [panel_balanced.data.sales[i]
                        for i in 1:length(panel_balanced.data.sales)
                        if panel_balanced.data.store[i] == "A"]
        @test count(ismissing, store_a_sales) == 2

        store_b_sales = [panel_balanced.data.sales[i]
                        for i in 1:length(panel_balanced.data.sales)
                        if panel_balanced.data.store[i] == "B"]
        @test count(ismissing, store_b_sales) == 1
    end

    @testset "Balanced panel validation" begin
        data = (
            store = ["A", "A"],
            date = [Date(2024,1,1), Date(2024,1,2)],
            sales = [100, 110]
        )

        @test_throws ArgumentError PanelData(data;
            groupby=:store,
            date=:date,
            frequency=:daily,
            fill_time=false,
            balanced=true
        )
    end

    @testset "resolve_m for single-season models" begin
        spec = ArimaSpec(Durbyn.Grammar.@formula(sales = p() + q()))

        @test Durbyn.ModelSpecs.resolve_m(nothing, spec) === nothing
        @test Durbyn.ModelSpecs.resolve_m(12, spec) == 12
        @test Durbyn.ModelSpecs.resolve_m([7, 365], spec) == 7
    end

    @testset "supports_multi_seasonality trait" begin
        arima_spec = ArimaSpec(Durbyn.Grammar.@formula(sales = p() + q()))
        @test Durbyn.ModelSpecs.supports_multi_seasonality(arima_spec) == false

        tbats_spec = TbatsSpec(Durbyn.Grammar.@formula(sales = tbats()))
        @test Durbyn.ModelSpecs.supports_multi_seasonality(tbats_spec) == true
    end

end
