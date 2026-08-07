import os
import pandas as pd
import numpy as np
from typing import Dict
from src.utils.logging_config import get_logger
from src.ingestion.clean_utils import (
    parse_dates,
    coerce_dtypes,
    handle_nulls,
    remove_duplicates,
    winsorize_sales
)

logger = get_logger("load_olist")


def load_olist_data(raw_data_dir: str = "data/raw/olist") -> Dict[str, pd.DataFrame]:
    """
    Ingests and transforms raw Olist Brazilian E-Commerce dataset into unified schema dataframes.
    Returns dict mapping table name -> pd.DataFrame.
    """
    logger.info(f"Loading Olist raw data from {raw_data_dir}...")

    # 1. Read Raw CSVs
    orders_path = os.path.join(raw_data_dir, "olist_orders_dataset.csv")
    items_path = os.path.join(raw_data_dir, "olist_order_items_dataset.csv")
    products_path = os.path.join(raw_data_dir, "olist_products_dataset.csv")
    sellers_path = os.path.join(raw_data_dir, "olist_sellers_dataset.csv")
    trans_path = os.path.join(raw_data_dir, "product_category_name_translation.csv")

    df_orders = pd.read_csv(orders_path)
    df_items = pd.read_csv(items_path)
    df_products = pd.read_csv(products_path)
    df_sellers = pd.read_csv(sellers_path)
    df_trans = pd.read_csv(trans_path) if os.path.exists(trans_path) else pd.DataFrame()

    # 2. Process product_dim
    if not df_trans.empty:
        df_products = df_products.merge(df_trans, on="product_category_name", how="left")
        df_products["product_name"] = df_products["product_category_name_english"].fillna(
            df_products["product_category_name"]
        ).fillna("Unknown Category")
    else:
        df_products["product_name"] = df_products["product_category_name"].fillna("Unknown Category")

    product_dim = pd.DataFrame({
        "product_id": "olist_prod_" + df_products["product_id"].astype(str),
        "product_name": df_products["product_name"].astype(str),
        "category": df_products["product_name"].astype(str).str.replace("_", " ").str.title(),
        "sub_category": df_products["product_category_name"].fillna("General").astype(str),
        "unit_cost": 0.0,
        "weight_g": df_products["product_weight_g"].fillna(0.0).astype(float),
        "dataset_source": "olist"
    })
    product_dim, _ = remove_duplicates(product_dim, subset=["product_id"])

    # 3. Process supplier_dim
    supplier_dim = pd.DataFrame({
        "supplier_id": "olist_supp_" + df_sellers["seller_id"].astype(str),
        "supplier_name": "Olist Seller " + df_sellers["seller_id"].astype(str).str[:8],
        "region": df_sellers["seller_state"].fillna("BR_UNKNOWN").astype(str),
        "city": df_sellers["seller_city"].fillna("Unknown").astype(str),
        "country": "Brazil",
        "lead_time_days": 5,
        "reliability_score": 4.5,
        "dataset_source": "olist"
    })
    supplier_dim, _ = remove_duplicates(supplier_dim, subset=["supplier_id"])

    # 4. Process warehouse_dim (group sellers by state/region)
    states = df_sellers["seller_state"].fillna("BR_UNKNOWN").unique()
    warehouse_rows = []
    for state in states:
        seller_in_state = df_sellers[df_sellers["seller_state"] == state]
        primary_city = seller_in_state["seller_city"].mode().iloc[0] if not seller_in_state.empty else "Unknown"
        warehouse_rows.append({
            "warehouse_id": f"olist_wh_{state}",
            "warehouse_name": f"Olist Fulfillment Hub ({state})",
            "region": state,
            "city": primary_city,
            "country": "Brazil",
            "capacity_units": 50000,
            "dataset_source": "olist"
        })
    warehouse_dim = pd.DataFrame(warehouse_rows)
    warehouse_dim, _ = remove_duplicates(warehouse_dim, subset=["warehouse_id"])

    # 5. Process promotion_dim
    promotion_dim = pd.DataFrame([{
        "promotion_id": "promo_none",
        "promo_name": "No Promotion",
        "discount_type": "none",
        "discount_percent": 0.0,
        "is_active": False,
        "dataset_source": "olist"
    }])

    # 6. Process sales_fact
    # Merge order items with orders and sellers
    df_sales_raw = df_items.merge(df_orders, on="order_id", how="inner")
    df_sales_raw = df_sales_raw.merge(df_sellers, on="seller_id", how="left")

    df_sales_raw = parse_dates(df_sales_raw, ["order_purchase_timestamp"])

    sales_fact = pd.DataFrame({
        "sales_id": "olist_" + df_sales_raw["order_id"].astype(str) + "_" + df_sales_raw["order_item_id"].astype(str),
        "dataset_source": "olist",
        "date": df_sales_raw["order_purchase_timestamp"],
        "product_id": "olist_prod_" + df_sales_raw["product_id"].astype(str),
        "warehouse_id": "olist_wh_" + df_sales_raw["seller_state"].fillna("BR_UNKNOWN").astype(str),
        "supplier_id": "olist_supp_" + df_sales_raw["seller_id"].astype(str),
        "promotion_id": "promo_none",
        "region": df_sales_raw["seller_state"].fillna("BR_UNKNOWN").astype(str),
        "quantity": 1.0,
        "unit_price": df_sales_raw["price"].astype(float),
        "total_sales": df_sales_raw["price"].astype(float),
        "discount_amount": 0.0,
        "shipping_cost": df_sales_raw["freight_value"].fillna(0.0).astype(float),
        "profit": 0.0
    })

    # Clean sales_fact
    sales_fact = sales_fact.dropna(subset=["date", "product_id"])
    sales_fact, _ = remove_duplicates(sales_fact, subset=["sales_id"])
    sales_fact = winsorize_sales(sales_fact, sales_col="total_sales")

    logger.info(
        f"Olist ingestion complete: sales_fact={len(sales_fact)}, product_dim={len(product_dim)}, "
        f"warehouse_dim={len(warehouse_dim)}, supplier_dim={len(supplier_dim)}"
    )

    return {
        "sales_fact": sales_fact,
        "product_dim": product_dim,
        "warehouse_dim": warehouse_dim,
        "supplier_dim": supplier_dim,
        "promotion_dim": promotion_dim
    }
