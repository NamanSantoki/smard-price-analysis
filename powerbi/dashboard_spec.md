# Power BI Dashboard Specification
## Wind, Solar and the German Day-Ahead Electricity Price

**Data source file:** `powerbi/dashboard_data.csv` (69,023 hourly rows, 24 columns)
**Data dictionary:** `powerbi/data_dictionary.json`

---

## Import Instructions

1. Open Power BI Desktop → Get Data → Text/CSV → select `dashboard_data.csv`.
2. In the Power Query Editor:
   - Change `datetime_berlin` column type to **Date/Time**.
   - Change `date` column type to **Date**.
   - All `_mw` columns → **Decimal Number**.
   - `price_eur_mwh` → **Decimal Number**.
   - Flag columns (`is_negative_price`, `is_weekend`, `is_holiday`, `is_workday`) → **Whole Number**.
3. Click **Close & Apply**.

---

## Suggested Slicers (apply to all pages)

| Slicer | Column | Type |
|--------|--------|------|
| Year | `year` | Dropdown or slider (2019–2026) |
| Month | `month` | Dropdown (1–12) |
| Hour of Day | `hour` | Slider (0–23) |
| Workday / Weekend | `is_workday` | Dropdown (0, 1) |

---

## Page 1: Overview – KPIs and Price Trends

### KPI Cards (top row)
| KPI | DAX Measure |
|-----|-------------|
| Average Price | `Avg Price = AVERAGE(dashboard_data[price_eur_mwh])` |
| Median Price | `Median Price = MEDIAN(dashboard_data[price_eur_mwh])` |
| Negative-Price Hours | `Neg Hours = SUM(dashboard_data[is_negative_price])` |
| Negative-Price Share | `Neg Share % = DIVIDE(SUM(dashboard_data[is_negative_price]), COUNTROWS(dashboard_data)) * 100` |
| Renewable Share of Load | `Renewable Share % = AVERAGE(dashboard_data[renewable_share_load]) * 100` |
| Total Hours | `Total Hours = COUNTROWS(dashboard_data)` |

### Visuals
1. **Line Chart**: Monthly average price over time (X: `date` by Month, Y: `Avg Price`).
2. **Bar Chart**: Annual negative-price hour count (X: `year`, Y: `Neg Hours`).
3. **Card with trend arrow**: Year-over-year change in average price.

### YoY DAX
```dax
Avg Price LY =
CALCULATE(
    [Avg Price],
    SAMEPERIODLASTYEAR('dashboard_data'[date])
)

YoY Change % =
DIVIDE([Avg Price] - [Avg Price LY], ABS([Avg Price LY])) * 100
```

---

## Page 2: Generation Mix & Price Drivers

### Visuals
1. **Stacked Area Chart**: Monthly generation by source.
   - X: `date` (Month), Y: `wind_onshore_mw`, `wind_offshore_mw`, `solar_mw`, `gas_mw`, `lignite_mw`, `hard_coal_mw`, `biomass_mw`, `hydro_mw`.
2. **Scatter Plot (Merit-Order Effect)**: X: `residual_load_mw` (binned or direct), Y: `price_eur_mwh`.
3. **Clustered Bar**: Average price by hour of day (X: `hour`, Y: `Avg Price`), coloured by weekday vs weekend.

### DAX Measures
```dax
Avg Wind Total GW = AVERAGE(dashboard_data[wind_total_mw]) / 1000
Avg Solar GW = AVERAGE(dashboard_data[solar_mw]) / 1000
Avg Load GW = AVERAGE(dashboard_data[load_mw]) / 1000
Avg Residual Load GW = AVERAGE(dashboard_data[residual_load_mw]) / 1000

Wind Share % =
DIVIDE(
    AVERAGE(dashboard_data[wind_total_mw]),
    AVERAGE(dashboard_data[load_mw])
) * 100

Solar Share % =
DIVIDE(
    AVERAGE(dashboard_data[solar_mw]),
    AVERAGE(dashboard_data[load_mw])
) * 100
```

---

## Page 3: Negative-Price Deep-Dive

### Visuals
1. **Heatmap / Matrix**: Negative-price share (%) by Month (rows) × Hour of Day (columns).
   - Rows: `month`, Columns: `hour`, Values: `Neg Share %`.
2. **Clustered Bar**: Negative-price hours per year (X: `year`, Y: `Neg Hours`), with data labels showing `Neg Share %`.
3. **Line + Area Combo**: On negative-price days, show average wind + solar generation vs load, to illustrate the oversupply.
4. **KPI Card**: Odds ratio of negative prices during high-solar hours (annotation from statistical testing).

### DAX Measures
```dax
Neg Hours by Month =
CALCULATE(
    SUM(dashboard_data[is_negative_price]),
    ALLEXCEPT(dashboard_data, dashboard_data[month])
)

Neg Share by Hour % =
DIVIDE(
    SUM(dashboard_data[is_negative_price]),
    COUNTROWS(dashboard_data)
) * 100

Avg Price Negative Hours =
CALCULATE(
    AVERAGE(dashboard_data[price_eur_mwh]),
    dashboard_data[is_negative_price] = 1
)

Avg Price Positive Hours =
CALCULATE(
    AVERAGE(dashboard_data[price_eur_mwh]),
    dashboard_data[is_negative_price] = 0
)
```

---

## Colour Palette Suggestion

| Element | Hex Code |
|---------|----------|
| Wind | `#1f77b4` (steel blue) |
| Solar | `#ff7f0e` (amber orange) |
| Price line | `#2ca02c` (green) |
| Negative price highlight | `#d62728` (crimson red) |
| Load | `#7f7f7f` (grey) |
| Fossil fuels (coal, gas) | `#8c564b`, `#9467bd` |

