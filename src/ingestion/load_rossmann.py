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

logger = get_logger("load_rossmann")


def load_rossmann_data(raw_data_dir: str = "data/raw/rossmann") -> Dict[str, pd.DataFrame]:
    """
    Ingests and transforms raw Rossmann Store Sales dataset into unified schema dataframes.
    Returns dict mapping table name -> pd.DataFrame.
    """
    logger.info(f"Loading Rossmann raw data from {raw_data_dir}...")

    train_path = os.path.join(raw_data_dir, "train.csv")
    store_path = os.path.join(raw_data_dir, "store.csv")

    df_train = pd.read_csv(train_path, dtype={"StateHoliday": str}, low_memory=False)
    df_store = pd.read_csv(store_path)

    # Merge train with store metadata
    df_merged = df_train.merge(df_store, on="Store", how="left")

    # 1. Process product_dim (1 catalog per store)
    product_dim = pd.DataFrame({
        "product_id": "rossmann_prod_store_" + df_store["Store"].astype(str),
        "product_name": "Rossmann Store Catalog " + df_store["Store"].astype(str),
        "category": "Pharmacy & Health",
        "sub_category": "Assortment " + df_store["Assortment"].fillna("Standard").astype(str),
        "unit_cost": 0.0,
        "weight_g": 0.0,
        "dataset_source": "rossmann"
    })
    product_dim, _ = remove_duplicates(product_dim, subset=["product_id"])

    # 2. Process warehouse_dim (Stores as retail warehouses)
    warehouse_dim = pd.DataFrame({
        "warehouse_id": "rossmann_wh_store_" + df_store["Store"].astype(str),
        "warehouse_name": "Rossmann Store " + df_store["Store"].astype(str),
        "region": "DE",
        "city": "Store Location " + df_store["Store"].astype(str),
        "country": "Germany",
        "capacity_units": 20000,
        "dataset_source": "rossmann"
    })
    warehouse_dim, _ = remove_duplicates(warehouse_dim, subset=["warehouse_id"])

    # 3. Process supplier_dim (Store Types acting as distribution hubs)
    store_types = df_store["StoreType"].dropna().unique()
    supplier_rows = []
    for stype in store_types:
        supplier_rows.append({
            "supplier_id": f"rossmann_supp_{stype}",
            "supplier_name": f"Rossmann Logistics Center ({stype})",
            "region": "DE",
            "city": "Central Logistics",
            "country": "Germany",
            "lead_time_days": 2,
            "reliability_score": 4.9,
            "dataset_source": "rossmann"
        })
    supplier_dim = pd.DataFrame(supplier_rows)
    supplier_dim, _ = remove_duplicates(supplier_dim, subset=["supplier_id"])

    # 4. Process promotion_dim
    promotion_dim = pd.DataFrame([{
        "promotion_id": "rossmann_promo_daily",
        "promo_name": "Rossmann Daily Store Promo",
        "discount_type": "flag",
        "discount_percent": 0.15,
        "is_active": True,
        "dataset_source": "rossmann"
    }])

    # 5. Process sales_fact
    # Filter for sales > 0 to keep active sales transactions
    df_active = df_merged[df_merged["Sales"] > 0].copy()
    df_active = parse_dates(df_active, ["Date"])

    promo_series = np.where(df_active["Promo"] == 1, "rossmann_promo_daily", "promo_none")

    # Estimate unit price = Sales / Customers if Customers > 0, else Sales
    cust_series = df_active["Customers"].clip(lower=1)
    unit_price_series = df_active["Sales"] / cust_series

    sales_fact = pd.DataFrame({
        "sales_id": "rossmann_" + df_active["Store"].astype(str) + "_" + df_active["Date"].astype(str),
        "dataset_source": "rossmann",
        "date": df_active["Date"],
        "product_id": "rossmann_prod_store_" + df_active["Store"].astype(str),
        "warehouse_id": "rossmann_wh_store_" + df_active["Store"].astype(str),
        "supplier_id": "rossmann_supp_" + df_active["StoreType"].fillna("a").astype(str),
        "promotion_id": promo_series,
        "region": "DE",
        "quantity": df_active["Customers"].astype(float),
        "unit_price": unit_price_series.astype(float),
        "total_sales": df_active["Sales"].astype(float),
        "discount_amount": 0.0,
        "shipping_cost": 0.0,
        "profit": 0.0
    })

    sales_fact = sales_fact.dropna(subset=["date", "product_id"])
    sales_fact, _ = remove_duplicates(sales_fact, subset=["sales_id"])
    sales_fact = winsorize_sales(sales_fact, sales_col="total_sales")

    logger.info(
        f"Rossmann ingestion complete: sales_fact={len(sales_fact)}, product_dim={len(product_dim)}, "
        f"warehouse_dim={len(warehouse_dim)}, supplier_dim={len(supplier_dim)}"
    )

    return {
        "sales_fact": sales_fact,
        "product_dim": product_dim,
        "warehouse_dim": warehouse_dim,
        "supplier_dim": supplier_dim,
        "promotion_dim": promotion_dim
    }
