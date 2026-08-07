# Pipeline Run Statistics Log

**Execution Timestamp**: `2026-07-31 16:12:33`  
**Total Run Duration**: `36.09 seconds`  
**Status**: `PASSED`  

## Summary Metrics per Processed Table

| Table Name | PK Column | Rows Input | Rows Output | Rows Rejected | Null Rate (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sales_fact` | `sales_id` | 1,143,942 | 1,143,942 | 0 | 0.000% | **PASSED** |
| `product_dim` | `product_id` | 34,229 | 34,229 | 0 | 0.000% | **PASSED** |
| `warehouse_dim` | `warehouse_id` | 1,206 | 1,206 | 0 | 0.000% | **PASSED** |
| `supplier_dim` | `supplier_id` | 3,111 | 3,111 | 0 | 0.000% | **PASSED** |
| `promotion_dim` | `promotion_id` | 4 | 4 | 0 | 0.000% | **PASSED** |
| `calendar_dim` | `date` | 3,133 | 3,133 | 0 | 0.000% | **PASSED** |
| `weather_dim` | `weather_id` | 11,161 | 11,161 | 0 | 0.000% | **PASSED** |
| `event_dim` | `event_id` | 156 | 156 | 0 | 0.000% | **PASSED** |

---
## Quality Checks & Validation Assertions
- **Primary Key Integrity**: 0 nulls detected across all tables.
- **Outlier Capping**: Sales winsorized at 1st and 99th percentiles.
- **Synthetic Metadata**: `is_synthetic=True` verified for `weather_dim` and `event_dim`.
- **Storage Format**: PyArrow Parquet exported to `data/processed/`.
