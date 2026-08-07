import os
import pandas as pd
import numpy as np
from typing import Dict
from src.utils.logging_config import get_logger
from src.ingestion.clean_utils import (
    parse_dates,
    remove_duplicates,
    winsorize_sales
)

logger = get_logger("load_m5")


def load_m5_data(raw_data_dir: str = "data/raw/m5") -> Dict[str, pd.DataFrame]:
    """
    Ingests and transforms raw M5 / Walmart dataset into unified schema dataframes.
    Returns dict mapping table name -> pd.DataFrame.
    """
    logger.info(f"Loading M5 raw data from {raw_data_dir}...")

    csv_path = os.path.join(raw_data_dir, "Walmart.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Walmart CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Process product_dim (1 aggregate catalog per Walmart store)
    stores = df["Store"].unique()
    product_rows = []
    for store_id in stores:
        product_rows.append({
            "product_id": f"m5_prod_store_{store_id}",
            "product_name": f"Walmart Store Catalog {store_id}",
            "category": "Retail Merchandise",
            "sub_category": "Grocery & General",
            "unit_cost": 0.0,
            "weight_g": 0.0,
            "dataset_source": "m5"
        })
    product_dim = pd.DataFrame(product_rows)
    product_dim, _ = remove_duplicates(product_dim, subset=["product_id"])

    # 2. Process warehouse_dim (Walmart store locations)
    warehouse_rows = []
    for store_id in stores:
        warehouse_rows.append({
            "warehouse_id": f"m5_wh_store_{store_id}",
            "warehouse_name": f"Walmart Supercenter {store_id}",
            "region": "US",
            "city": f"Walmart Location {store_id}",
            "country": "USA",
            "capacity_units": 150000,
            "dataset_source": "m5"
        })
    warehouse_dim = pd.DataFrame(warehouse_rows)
    warehouse_dim, _ = remove_duplicates(warehouse_dim, subset=["warehouse_id"])

    # 3. Process supplier_dim
    supplier_dim = pd.DataFrame([{
        "supplier_id": "m5_supp_walmart",
        "supplier_name": "Walmart Global Distribution Network",
        "region": "US",
        "city": "Bentonville",
        "country": "USA",
        "lead_time_days": 4,
        "reliability_score": 4.7,
        "dataset_source": "m5"
    }])

    # 4. Process promotion_dim
    promotion_dim = pd.DataFrame([{
        "promotion_id": "m5_promo_holiday",
        "promo_name": "Walmart Holiday Event",
        "discount_type": "flag",
        "discount_percent": 0.10,
        "is_active": True,
        "dataset_source": "m5"
    }])

    # 5. Process sales_fact
    # Date in Walmart.csv is DD-MM-YYYY format
    df = parse_dates(df, ["Date"], date_format="%d-%m-%Y")

    promo_series = np.where(df["Holiday_Flag"] == 1, "m5_promo_holiday", "promo_none")

    sales_fact = pd.DataFrame({
        "sales_id": "m5_" + df["Store"].astype(str) + "_" + df["Date"].astype(str),
        "dataset_source": "m5",
        "date": df["Date"],
        "product_id": "m5_prod_store_" + df["Store"].astype(str),
        "warehouse_id": "m5_wh_store_" + df["Store"].astype(str),
        "supplier_id": "m5_supp_walmart",
        "promotion_id": promo_series,
        "region": "US",
        "quantity": 1.0,
        "unit_price": df["Weekly_Sales"].astype(float),
        "total_sales": df["Weekly_Sales"].astype(float),
        "discount_amount": 0.0,
        "shipping_cost": 0.0,
        "profit": 0.0
    })

    sales_fact = sales_fact.dropna(subset=["date", "product_id"])
    sales_fact, _ = remove_duplicates(sales_fact, subset=["sales_id"])
    sales_fact = winsorize_sales(sales_fact, sales_col="total_sales")

    logger.info(
        f"M5 ingestion complete: sales_fact={len(sales_fact)}, product_dim={len(product_dim)}, "
        f"warehouse_dim={len(warehouse_dim)}, supplier_dim={len(supplier_dim)}"
    )

    return {
        "sales_fact": sales_fact,
        "product_dim": product_dim,
        "warehouse_dim": warehouse_dim,
        "supplier_dim": supplier_dim,
        "promotion_dim": promotion_dim
    }
