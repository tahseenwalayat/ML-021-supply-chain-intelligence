# Phase 12: Supply Chain Risk Engine

## Overview

The **Supply Chain Risk Engine** is a core operational module of the Enterprise Supply Chain Demand Forecasting & Inventory Optimization Platform. It continuously evaluates multi-dimensional operational risks across inventory nodes, SKUs, and suppliers to prevent stockouts, eliminate tied-up working capital in dead stock, detect demand anomalies, and mitigate supplier lead time disruptions.

---

## Architecture & 5-Dimensional Risk Framework

The engine computes a bounded composite risk score $S_{\text{composite}} \in [0.0, 1.0]$ by aggregating five specialized risk dimensions:

$$\text{Composite Risk Score} = w_1 S_{\text{supplier}} + w_2 S_{\text{stockout}} + w_3 S_{\text{overstock}} + w_4 S_{\text{health}} + w_5 S_{\text{anomaly}}$$

Where default normalized weights are:
- **Supplier Delay Risk ($w_1 = 0.25$)**
- **Stockout Risk ($w_2 = 0.30$)**
- **Overstock Risk ($w_3 = 0.15$)**
- **Inventory Health Risk ($w_4 = 0.15$)**
- **Demand Anomaly Risk ($w_5 = 0.15$)**

---

## 1. Risk Dimensions & Mathematical Formulations

### 1.1 Supplier Delay Risk (`src/risk/supplier_delay_risk.py`)
Evaluates supplier delivery variance and historical reliability.
- **Inputs**: Supplier Reliability Score ($R \in [0, 1]$), Lead Time Std Dev ($\sigma_L$), Mean Lead Time ($\bar{L}$), Late Delivery Rate ($L_{\text{late}}$).
- **Formulation**:
  $$S_{\text{supplier}} = w_{\text{late}} \cdot L_{\text{late}} + w_{\text{var}} \cdot \min\left(1.0, \frac{\sigma_L}{\bar{L}}\right)$$
- **Buffer Recommendation**: Calculates required safety buffer days $\text{Buffer} = 1.96 \cdot \sigma_L$.

### 1.2 Stockout Risk (`src/risk/stockout_risk.py`)
Evaluates inventory exhaustion vulnerability prior to replenishment arrival.
- **Inputs**: Current Stock ($I$), Reorder Point ($ROP$), Safety Stock ($SS$), Average Daily Demand ($\bar{d}$), Lead Time ($L$).
- **Formulation**:
  - Out of stock ($I \le 0$): $S_{\text{stockout}} = 1.0$
  - Safety stock breached ($I \le SS$): $S_{\text{stockout}} = 0.75 + 0.25 \cdot \left(1 - \frac{I}{SS}\right)$
  - ROP breached ($I \le ROP$): $S_{\text{stockout}} = 0.25 + 0.50 \cdot \frac{ROP - I}{ROP - SS}$
  - Healthy ($I > ROP$): $S_{\text{stockout}} = \max\left(0.0, 0.25 \cdot \left(1 - \frac{I - ROP}{ROP}\right)\right)$

### 1.3 Overstock Risk (`src/risk/overstock_risk.py`)
Identifies excessive stock carrying levels that tie up working capital.
- **Inputs**: Current Stock ($I$), Reorder Point ($ROP$), Unit Cost ($C$), Overstock Multiplier $M_{\text{rop}} = 3.0$.
- **Formulation**:
  - Excess Units: $I_{\text{excess}} = \max(0.0, I - M_{\text{rop}} \cdot ROP)$
  - Tied-up Capital: $\text{Capital} = I_{\text{excess}} \cdot C$
  - Risk Score: Exponential ramp based on excess ratio.

### 1.4 Inventory Health Risk / Slow & Dead Stock (`src/risk/inventory_health_risk.py`, `src/risk/slow_dead_inventory.py`)
Detects stagnating inventory and obsolete SKUs.
- **Categories**:
  - **DEAD STOCK**: $12+$ consecutive weeks of zero sales ($S_{\text{health}} \ge 0.75$).
  - **SLOW MOVING**: Sales velocity $< 1.0$ unit/day ($S_{\text{health}} \in [0.40, 0.75)$).
  - **HEALTHY**: Active sales velocity ($S_{\text{health}} < 0.40$).

### 1.5 Demand Anomaly Risk (`src/risk/demand_anomaly_risk.py`, `src/risk/anomaly_detection.py`)
Monitors demand volatility and flags statistical anomalies.
- **Inputs**: Observed Demand ($d$), Mean Demand ($\mu_d$), Demand Std Dev ($\sigma_d$).
- **Formulation**:
  $$Z = \frac{d - \mu_d}{\sigma_d}$$
  - Flagged as **DEMAND_SPIKE** if $Z \ge +3.0$.
  - Flagged as **DEMAND_DROP** if $Z \le -3.0$.

---

## 2. Risk Classification Levels

| Score Range | Risk Level | Action Required |
| :--- | :--- | :--- |
| $S \ge 0.75$ | **CRITICAL** | Immediate emergency intervention (purchase order expedite, inter-warehouse transfer, clearance liquidation) |
| $0.50 \le S < 0.75$ | **HIGH** | Priority review by procurement & logistics managers |
| $0.25 \le S < 0.50$ | **MEDIUM** | Routine replenishment adjustments & buffer monitoring |
| $S < 0.25$ | **LOW** | Operational parameters within normal tolerances |

---

## 3. FastAPI REST Endpoints (`src/api/risk_router.py`)

- `POST /api/v1/risk/supplier-delay` - Evaluates supplier delay risk metrics and buffer days.
- `POST /api/v1/risk/stockout` - Evaluates stockout vulnerability and breach status.
- `POST /api/v1/risk/overstock` - Evaluates overstock exposure and tied-up working capital.
- `POST /api/v1/risk/inventory-health` - Evaluates slow-moving and dead inventory classification.
- `POST /api/v1/risk/demand-anomaly` - Evaluates Z-score demand anomalies.
- `POST /api/v1/risk/evaluate-item` - Evaluates full 5D composite risk for a single SKU/warehouse pair.
- `POST /api/v1/risk/evaluate-batch` - Vectorized evaluation for a batch of SKUs with summary scorecards.

---

## 4. Pipeline Execution & Verification

### Pipeline Runner
Execute batch risk calculation across processed inventory datasets:
```bash
python -m src.risk.run_risk_engine
```
Generates output artifact: `data/processed/risk_scores.parquet`.

### Automated Verification Tests
Run unit and integration test suite:
```bash
pytest tests/test_risk_engine.py tests/test_supplier_delay_risk.py tests/test_risk_scores.py -v
```
All test suites pass with 100% coverage across risk models, edge cases, and API endpoints.
