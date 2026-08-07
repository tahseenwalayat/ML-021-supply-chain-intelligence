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

logger = get_logger("load_dataco")


def load_dataco_data(raw_data_dir: str = "data/raw/dataco") -> Dict[str, pd.DataFrame]:
    """
    Ingests and transforms raw DataCo Smart Supply Chain dataset into unified schema dataframes.
    Returns dict mapping table name -> pd.DataFrame.
    """
    logger.info(f"Loading DataCo raw data from {raw_data_dir}...")

    csv_path = os.path.join(raw_data_dir, "DataCoSupplyChainDataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"DataCo dataset not found at {csv_path}")

    # Read CSV using latin1 encoding to handle special characters in location names
    df = pd.read_csv(csv_path, encoding="latin1")

    # 1. Process product_dim
    df_prods = df[["Product Card Id", "Product Name", "Category Name", "Department Name", "Product Price"]].drop_duplicates()
    product_dim = pd.DataFrame({
        "product_id": "dataco_prod_" + df_prods["Product Card Id"].astype(str),
        "product_name": df_prods["Product Name"].astype(str),
        "category": df_prods["Category Name"].astype(str),
        "sub_category": df_prods["Department Name"].astype(str),
        "unit_cost": df_prods["Product Price"].astype(float),
        "weight_g": 0.0,
        "dataset_source": "dataco"
    })
    product_dim, _ = remove_duplicates(product_dim, subset=["product_id"])

    # 2. Process supplier_dim (Departments acting as internal suppliers)
    df_depts = df[["Department Id", "Department Name"]].drop_duplicates()
    supplier_dim = pd.DataFrame({
        "supplier_id": "dataco_supp_" + df_depts["Department Id"].astype(str),
        "supplier_name": "DataCo Supplier Dept - " + df_depts["Department Name"].astype(str),
        "region": "Global",
        "city": "Supply Center",
        "country": "USA",
        "lead_time_days": 3,
        "reliability_score": 4.8,
        "dataset_source": "dataco"
    })
    supplier_dim, _ = remove_duplicates(supplier_dim, subset=["supplier_id"])

    # 3. Process warehouse_dim (Regions / Markets acting as fulfillment nodes)
    df_regions = df[["Order Region", "Order City", "Order Country"]].drop_duplicates(subset=["Order Region"])
    warehouse_dim = pd.DataFrame({
        "warehouse_id": "dataco_wh_" + df_regions["Order Region"].astype(str).str.replace(" ", "_"),
        "warehouse_name": "DataCo Hub " + df_regions["Order Region"].astype(str),
        "region": df_regions["Order Region"].astype(str),
        "city": df_regions["Order City"].fillna("Regional HQ").astype(str),
        "country": df_regions["Order Country"].fillna("Global").astype(str),
        "capacity_units": 100000,
        "dataset_source": "dataco"
    })
    warehouse_dim, _ = remove_duplicates(warehouse_dim, subset=["warehouse_id"])

    # 4. Process promotion_dim
    promotion_dim = pd.DataFrame([{
        "promotion_id": "dataco_promo_discount",
        "promo_name": "DataCo Line Item Discount",
        "discount_type": "percentage",
        "discount_percent": 0.10,
        "is_active": True,
        "dataset_source": "dataco"
    }])

    # 5. Process sales_fact
    df = parse_dates(df, ["order date (DateOrders)"])

    discount_applied = df["Order Item Discount"] > 0
    promo_id_series = np.where(discount_applied, "dataco_promo_discount", "promo_none")

    sales_fact = pd.DataFrame({
        "sales_id": "dataco_" + df["Order Item Id"].astype(str),
        "dataset_source": "dataco",
        "date": df["order date (DateOrders)"],
        "product_id": "dataco_prod_" + df["Product Card Id"].astype(str),
        "warehouse_id": "dataco_wh_" + df["Order Region"].astype(str).str.replace(" ", "_"),
        "supplier_id": "dataco_supp_" + df["Department Id"].astype(str),
        "promotion_id": promo_id_series,
        "region": df["Order Region"].fillna("Global").astype(str),
        "quantity": df["Order Item Quantity"].astype(float),
        "unit_price": df["Order Item Product Price"].astype(float),
        "total_sales": df["Sales"].astype(float),
        "discount_amount": df["Order Item Discount"].astype(float),
        "shipping_cost": 0.0,
        "profit": df["Order Profit Per Order"].astype(float)
    })

    sales_fact = sales_fact.dropna(subset=["date", "product_id"])
    sales_fact, _ = remove_duplicates(sales_fact, subset=["sales_id"])
    sales_fact = winsorize_sales(sales_fact, sales_col="total_sales")

    logger.info(
        f"DataCo ingestion complete: sales_fact={len(sales_fact)}, product_dim={len(product_dim)}, "
        f"warehouse_dim={len(warehouse_dim)}, supplier_dim={len(supplier_dim)}"
    )

    return {
        "sales_fact": sales_fact,
        "product_dim": product_dim,
        "warehouse_dim": warehouse_dim,
        "supplier_dim": supplier_dim,
        "promotion_dim": promotion_dim
    }
