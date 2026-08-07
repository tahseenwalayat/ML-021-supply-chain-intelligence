# Phase 13: Scenario Simulation Engine

## Overview

The **Scenario Simulation Engine** provides enterprise supply chain decision-makers with interactive and batch stress-testing capabilities. It simulates dynamic operational shocks (supplier failures, price adjustments, holiday peaks, new product launches, transport delays, and demand surges) and computes projected before/after operational impacts before real-world disruptions materialize.

---

## 6 Scenario Types & Architectural Mapping

All scenario types derive from the abstract [`BaseScenario`](file:///D:/ML-021/ML-021/src/simulation/base.py) interface and reuse existing forecasting, inventory allocation, and risk evaluation engines.

### 1. Supplier Failure Scenario ([`src/simulation/supplier_failure.py`](file:///D:/ML-021/ML-021/src/simulation/supplier_failure.py))
- **YAML Config**: [`configs/scenarios/01_supplier_failure.yaml`](file:///D:/ML-021/ML-021/configs/scenarios/01_supplier_failure.yaml)
- **Perturbation**: Increases lead time by $N$ days (e.g., $+14$ days) and lead time variance ($\times 2.5$) for target suppliers.
- **Impact**: Recomputes stockout risk escalation, stockout exposure days, and required safety stock buffers.

### 2. Price Increase Scenario ([`src/simulation/price_increase.py`](file:///D:/ML-021/ML-021/src/simulation/price_increase.py))
- **YAML Config**: [`configs/scenarios/02_price_increase.yaml`](file:///D:/ML-021/ML-021/configs/scenarios/02_price_increase.yaml)
- **Perturbation**: Applies price change percentage ($\Delta P\%$) with price elasticity of demand ($E_{\text{price}}$):
  $$\Delta D\% = E_{\text{price}} \cdot \Delta P\%$$
- **Impact**: Adjusts sales velocity, expected demand, and evaluates revenue vs overstock risk changes.

### 3. Holiday Sales Scenario ([`src/simulation/holiday_sales.py`](file:///D:/ML-021/ML-021/src/simulation/holiday_sales.py))
- **YAML Config**: [`configs/scenarios/03_holiday_sales.yaml`](file:///D:/ML-021/ML-021/configs/scenarios/03_holiday_sales.yaml)
- **Perturbation**: Applies multiplicative demand multipliers (e.g., $1.75\times$) during peak holiday/Cyber Week windows.
- **Impact**: Projects fill-rate degradation and identifies potential stockout bottlenecks.

### 4. New Product Launch Scenario ([`src/simulation/new_product_launch.py`](file:///D:/ML-021/ML-021/src/simulation/new_product_launch.py))
- **YAML Config**: [`configs/scenarios/04_new_product_launch.yaml`](file:///D:/ML-021/ML-021/configs/scenarios/04_new_product_launch.yaml)
- **Perturbation**: Uses category-average demand profiles as a cold-start proxy for new SKUs with ramp-up curves over 14 days.
- **Impact**: Establishes initial safety stock targets, reorder points, and procurement capital needs.

### 5. Transport Delay Scenario ([`src/simulation/transport_delay.py`](file:///D:/ML-021/ML-021/src/simulation/transport_delay.py))
- **YAML Config**: [`configs/scenarios/05_transport_delay.yaml`](file:///D:/ML-021/ML-021/configs/scenarios/05_transport_delay.yaml)
- **Perturbation**: Increases lead time by $N$ days (e.g., $+6$ days) across all SKUs in a target geographical region.
- **Impact**: Evaluates regional reorder point shifts and stockout risk deltas.

### 6. Demand Surge Scenario ([`src/simulation/demand_surge.py`](file:///D:/ML-021/ML-021/src/simulation/demand_surge.py))
- **YAML Config**: [`configs/scenarios/06_demand_surge.yaml`](file:///D:/ML-021/ML-021/configs/scenarios/06_demand_surge.yaml)
- **Perturbation**: Applies multiplicative macroeconomic demand shock (e.g., $1.50\times$) across nodes.
- **Impact**: Reruns inventory allocation, fill rates, and extra capital requirements.

---

## Before / After Output Format (`data/processed/scenario_comparison_report.parquet`)

The [`ScenarioOrchestrator`](file:///D:/ML-021/ML-021/src/simulation/orchestrator.py) compiles comparative metrics into a standardized report:

| Attribute | Description | Example |
| :--- | :--- | :--- |
| `scenario_name` | Name of the scenario run | `"Supplier Failure & Severe Lead Time Spike"` |
| `scenario_type` | Scenario identifier | `"supplier_failure"` |
| `horizon_days` | Simulation horizon (days) | `30` |
| `baseline_fill_rate_pct` | Mean fill rate before shock (%) | `98.42%` |
| `scenario_fill_rate_pct` | Mean fill rate after shock (%) | `72.50%` |
| `fill_rate_delta_pct` | Fill rate change percentage points | `-25.92%` |
| `baseline_stockout_risk` | Mean baseline stockout risk score | `0.1245` |
| `scenario_stockout_risk` | Mean scenario stockout risk score | `0.6840` |
| `stockout_risk_delta` | Stockout risk score increase | `+0.5595` |
| `stockout_days_increase` | Total cumulative stockout days increase | `+480 days` |
| `extra_capital_required_usd` | Additional working capital required ($) | `$3,161.08` |

---

## Pipeline Execution & Verification

### Run Orchestrator Pipeline
```bash
python -m src.simulation.orchestrator
```
Generates artifact: `data/processed/scenario_comparison_report.parquet`.

### Run Test Suite
```bash
pytest tests/test_scenarios.py -v
```
Validates all 6 scenario types, single-product supplier edge cases, and 0% price change sanity checks.
