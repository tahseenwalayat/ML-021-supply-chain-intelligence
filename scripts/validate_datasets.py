#!/usr/bin/env python3
"""
Dataset Validation and Profiling Script for Supply Chain Forecasting Platform.

Reads raw datasets from data/raw/ (olist, dataco, rossmann, m5), inspects schemas,
missingness, and date ranges without mutating data, and outputs a formatted Markdown
report to docs/dataset_profile.md.
"""

import os
import glob
import pandas as pd

# Define dataset folders under data/raw/ with fallback to Datasets/ if needed
DATASET_MAPPING = {
    'olist': ['data/raw/olist', 'Datasets/Brazilian E-Commerce Public Dataset by Olist'],
    'dataco': ['data/raw/dataco', 'Datasets/DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS'],
    'rossmann': ['data/raw/rossmann', 'Datasets/Kaggle-Rossman-Sales-Prediction'],
    'm5': ['data/raw/m5', 'Datasets/M5 Forecasting - Accuracy']
}

OUTPUT_MARKDOWN_PATH = 'docs/dataset_profile.md'


def find_csv_files(folder_candidates):
    """Locate all CSV files in candidate directory paths."""
    for folder in folder_candidates:
        if os.path.exists(folder):
            csvs = sorted(glob.glob(os.path.join(folder, '*.csv')))
            if csvs:
                return csvs, folder
    return [], None


def profile_dataframe(csv_path):
    """Load CSV safely and return profiling metadata."""
    fname = os.path.basename(csv_path)
    
    # Try reading UTF-8 first, fallback to latin1 / ISO-8859-1 if needed
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)

    num_rows, num_cols = df.shape
    col_profiles = []

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        null_pct = (null_count / num_rows * 100) if num_rows > 0 else 0.0
        
        min_date = "N/A"
        max_date = "N/A"
        
        # Check if column is a potential date column
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'year', 'month']) or col_lower == 'date':
            try:
                # Attempt to parse non-null values as datetime
                non_null_vals = df[col].dropna()
                if len(non_null_vals) > 0:
                    dt_series = pd.to_datetime(non_null_vals, errors='coerce')
                    valid_dts = dt_series.dropna()
                    if len(valid_dts) > 0:
                        min_date = str(valid_dts.min())
                        max_date = str(valid_dts.max())
            except Exception:
                pass
        
        col_profiles.append({
            'col_name': col,
            'dtype': dtype_str,
            'null_count': null_count,
            'null_pct': f"{null_pct:.2f}%",
            'min_date': min_date,
            'max_date': max_date
        })

    return {
        'filename': fname,
        'path': csv_path,
        'rows': num_rows,
        'cols': num_cols,
        'columns': col_profiles
    }


def generate_markdown_report(profile_results):
    """Generate structured Markdown document content."""
    lines = []
    lines.append("# Raw Dataset Validation & Schema Profiling Report\n")
    lines.append("## Overview\n")
    lines.append("This document presents a comprehensive, read-only profiling summary of the 4 raw supply chain benchmark datasets: ")
    lines.append("**Olist (Brazilian E-Commerce)**, **DataCo Smart Supply Chain**, **Rossmann Store Sales**, and **M5 Forecasting (Walmart)**. ")
    lines.append("For each dataset file, schema dimensions, column data types, missing value percentages, and detected date ranges are documented.\n")
    lines.append("---\n")

    for ds_name, files_info in profile_results.items():
        lines.append(f"## Dataset: {ds_name.upper()}\n")
        lines.append(f"**Total Files**: {len(files_info)}\n")

        for info in files_info:
            lines.append(f"### File: `{info['filename']}`")
            lines.append(f"- **Dimensions**: `{info['rows']:,}` rows x `{info['cols']}` columns")
            lines.append("- **Column Schema & Missingness Analysis**:\n")

            lines.append("| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for col in info['columns']:
                lines.append(
                    f"| `{col['col_name']}` | `{col['dtype']}` | `{col['null_count']:,}` | `{col['null_pct']}` | `{col['min_date']}` | `{col['max_date']}` |"
                )
            lines.append("\n")

        lines.append("---\n")

    # Add Candidate Primary Keys and Join Keys
    lines.append("## Candidate Primary Keys & Join Keys per Dataset\n")

    lines.append("### 1. Olist Brazilian E-Commerce Dataset")
    lines.append("- **Primary Keys**:")
    lines.append("  - `olist_customers_dataset.csv`: `customer_id` (Primary Key, unique per order instance)")
    lines.append("  - `olist_geolocation_dataset.csv`: `geolocation_zip_code_prefix` (Non-unique location index)")
    lines.append("  - `olist_order_items_dataset.csv`: Composite (`order_id`, `order_item_id`)")
    lines.append("  - `olist_order_payments_dataset.csv`: Composite (`order_id`, `payment_sequential`)")
    lines.append("  - `olist_order_reviews_dataset.csv`: `review_id` (Composite: `review_id`, `order_id`)")
    lines.append("  - `olist_orders_dataset.csv`: `order_id` (Primary Key)")
    lines.append("  - `olist_products_dataset.csv`: `product_id` (Primary Key)")
    lines.append("  - `olist_sellers_dataset.csv`: `seller_id` (Primary Key)")
    lines.append("  - `product_category_name_translation.csv`: `product_category_name` (Primary Key)")
    lines.append("- **Join Keys & Foreign Relationships**:")
    lines.append("  - `order_id`: Links `olist_orders_dataset` to `order_items`, `order_payments`, and `order_reviews`")
    lines.append("  - `customer_id`: Links `olist_orders_dataset` to `olist_customers_dataset` (`customer_id` connects to `customer_unique_id`)")
    lines.append("  - `product_id`: Links `olist_order_items_dataset` to `olist_products_dataset`")
    lines.append("  - `seller_id`: Links `olist_order_items_dataset` to `olist_sellers_dataset`")
    lines.append("  - `product_category_name`: Links `olist_products_dataset` to `product_category_name_translation`")
    lines.append("  - `zip_code_prefix`: Links `customers`/`sellers` to `olist_geolocation_dataset`\n")

    lines.append("### 2. DataCo Smart Supply Chain Dataset")
    lines.append("- **Primary Keys**:")
    lines.append("  - `DataCoSupplyChainDataset.csv`: `Order Item Id` (Primary Key per row), composite (`Order Id`, `Order Item Id`)")
    lines.append("  - `DescriptionDataCoSupplyChain.csv`: `FIELDS` (Primary Key for field metadata)")
    lines.append("  - `tokenized_access_logs.csv`: Transactional event log (No strict primary key; composite timestamp + IP + URL)")
    lines.append("- **Join Keys & Foreign Relationships**:")
    lines.append("  - `Order Id` / `Order Item Id`: Connects order lines to shipment tracking and financial calculation logic")
    lines.append("  - `Customer Id` / `Order Customer Id`: Links customers to order transactions")
    lines.append("  - `Product Card Id` / `Category Id` / `Department Id`: Connects order lines to product master catalog and organizational hierarchy\n")

    lines.append("### 3. Rossmann Store Sales Dataset")
    lines.append("- **Primary Keys**:")
    lines.append("  - `store.csv`: `Store` (Primary Key)")
    lines.append("  - `train.csv`: Composite (`Store`, `Date`)")
    lines.append("  - `test.csv`: `Id` (Primary Key), composite (`Store`, `Date`)")
    lines.append("  - `sample_submission.csv`: `Id` (Primary Key)")
    lines.append("- **Join Keys & Foreign Relationships**:")
    lines.append("  - `Store`: Primary foreign key joining `train.csv` and `test.csv` to `store.csv` metadata")
    lines.append("  - `Date`: Time join key for joining seasonal external calendars and promotional schedules\n")

    lines.append("### 4. M5 Forecasting / Walmart Dataset")
    lines.append("- **Primary Keys**:")
    lines.append("  - `Walmart.csv`: Composite (`Store`, `Date`)")
    lines.append("- **Join Keys & Foreign Relationships**:")
    lines.append("  - `Store`: Joins store location attributes and regional parameters")
    lines.append("  - `Date`: Time-series join key for macroeconomic indicators (CPI, Fuel Price, Unemployment, Temperature)\n")

    return "\n".join(lines)


def main():
    print("Starting raw dataset validation and profiling...")
    profile_results = {}

    for ds_name, candidates in DATASET_MAPPING.items():
        csv_files, active_folder = find_csv_files(candidates)
        print(f"\nProcessing Dataset [{ds_name.upper()}] from '{active_folder}'...")
        if not csv_files:
            print(f"  Warning: No CSV files found for {ds_name}!")
            continue

        ds_profiles = []
        for csv_path in csv_files:
            fname = os.path.basename(csv_path)
            print(f"  Profiling '{fname}'...")
            info = profile_dataframe(csv_path)
            ds_profiles.append(info)

        profile_results[ds_name] = ds_profiles

    # Create docs directory if missing
    os.makedirs(os.path.dirname(OUTPUT_MARKDOWN_PATH), exist_ok=True)

    # Generate Markdown report
    markdown_content = generate_markdown_report(profile_results)

    with open(OUTPUT_MARKDOWN_PATH, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"\nValidation complete! Profiling report written to '{OUTPUT_MARKDOWN_PATH}'.")


if __name__ == '__main__':
    main()
