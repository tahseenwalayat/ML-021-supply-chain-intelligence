# Raw Dataset Validation & Schema Profiling Report

## Overview

This document presents a comprehensive, read-only profiling summary of the 4 raw supply chain benchmark datasets: 
**Olist (Brazilian E-Commerce)**, **DataCo Smart Supply Chain**, **Rossmann Store Sales**, and **M5 Forecasting (Walmart)**. 
For each dataset file, schema dimensions, column data types, missing value percentages, and detected date ranges are documented.

---

## Dataset: OLIST

**Total Files**: 9

### File: `olist_customers_dataset.csv`
- **Dimensions**: `99,441` rows x `5` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customer_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `customer_unique_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `customer_zip_code_prefix` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `customer_city` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `customer_state` | `str` | `0` | `0.00%` | `N/A` | `N/A` |


### File: `olist_geolocation_dataset.csv`
- **Dimensions**: `1,000,163` rows x `5` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `geolocation_zip_code_prefix` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `geolocation_lat` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `geolocation_lng` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `geolocation_city` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `geolocation_state` | `str` | `0` | `0.00%` | `N/A` | `N/A` |


### File: `olist_order_items_dataset.csv`
- **Dimensions**: `112,650` rows x `7` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `order_item_id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `product_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `seller_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `shipping_limit_date` | `str` | `0` | `0.00%` | `2016-09-19 00:15:34` | `2020-04-09 22:35:08` |
| `price` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `freight_value` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |


### File: `olist_order_payments_dataset.csv`
- **Dimensions**: `103,886` rows x `5` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `payment_sequential` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `payment_type` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `payment_installments` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `payment_value` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |


### File: `olist_order_reviews_dataset.csv`
- **Dimensions**: `99,224` rows x `7` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `review_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `order_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `review_score` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `review_comment_title` | `str` | `87,656` | `88.34%` | `N/A` | `N/A` |
| `review_comment_message` | `str` | `58,247` | `58.70%` | `N/A` | `N/A` |
| `review_creation_date` | `str` | `0` | `0.00%` | `2016-10-02 00:00:00` | `2018-08-31 00:00:00` |
| `review_answer_timestamp` | `str` | `0` | `0.00%` | `2016-10-07 18:32:28` | `2018-10-29 12:27:35` |


### File: `olist_orders_dataset.csv`
- **Dimensions**: `99,441` rows x `8` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `customer_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `order_status` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `order_purchase_timestamp` | `str` | `0` | `0.00%` | `2016-09-04 21:15:19` | `2018-10-17 17:30:18` |
| `order_approved_at` | `str` | `160` | `0.16%` | `N/A` | `N/A` |
| `order_delivered_carrier_date` | `str` | `1,783` | `1.79%` | `2016-10-08 10:34:01` | `2018-09-11 19:48:28` |
| `order_delivered_customer_date` | `str` | `2,965` | `2.98%` | `2016-10-11 13:46:32` | `2018-10-17 13:22:46` |
| `order_estimated_delivery_date` | `str` | `0` | `0.00%` | `2016-09-30 00:00:00` | `2018-11-12 00:00:00` |


### File: `olist_products_dataset.csv`
- **Dimensions**: `32,951` rows x `9` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `product_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `product_category_name` | `str` | `610` | `1.85%` | `N/A` | `N/A` |
| `product_name_lenght` | `float64` | `610` | `1.85%` | `N/A` | `N/A` |
| `product_description_lenght` | `float64` | `610` | `1.85%` | `N/A` | `N/A` |
| `product_photos_qty` | `float64` | `610` | `1.85%` | `N/A` | `N/A` |
| `product_weight_g` | `float64` | `2` | `0.01%` | `N/A` | `N/A` |
| `product_length_cm` | `float64` | `2` | `0.01%` | `N/A` | `N/A` |
| `product_height_cm` | `float64` | `2` | `0.01%` | `N/A` | `N/A` |
| `product_width_cm` | `float64` | `2` | `0.01%` | `N/A` | `N/A` |


### File: `olist_sellers_dataset.csv`
- **Dimensions**: `3,095` rows x `4` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `seller_id` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `seller_zip_code_prefix` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `seller_city` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `seller_state` | `str` | `0` | `0.00%` | `N/A` | `N/A` |


### File: `product_category_name_translation.csv`
- **Dimensions**: `71` rows x `2` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `product_category_name` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `product_category_name_english` | `str` | `0` | `0.00%` | `N/A` | `N/A` |


---

## Dataset: DATACO

**Total Files**: 3

### File: `DataCoSupplyChainDataset.csv`
- **Dimensions**: `180,519` rows x `53` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Type` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Days for shipping (real)` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Days for shipment (scheduled)` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Benefit per order` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Sales per customer` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Delivery Status` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Late_delivery_risk` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Category Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Category Name` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customer City` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customer Country` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customer Email` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customer Fname` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customer Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customer Lname` | `str` | `8` | `0.00%` | `N/A` | `N/A` |
| `Customer Password` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customer Segment` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customer State` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customer Street` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customer Zipcode` | `float64` | `3` | `0.00%` | `N/A` | `N/A` |
| `Department Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Department Name` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Latitude` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Longitude` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Market` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order City` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Country` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Customer Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `order date (DateOrders)` | `str` | `0` | `0.00%` | `2015-01-01 00:00:00` | `2018-01-31 23:38:00` |
| `Order Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Item Cardprod Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Item Discount` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Item Discount Rate` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Item Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Item Product Price` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Item Profit Ratio` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Item Quantity` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Sales` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Item Total` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Profit Per Order` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Region` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order State` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Status` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Order Zipcode` | `float64` | `155,679` | `86.24%` | `N/A` | `N/A` |
| `Product Card Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Product Category Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Product Description` | `float64` | `180,519` | `100.00%` | `N/A` | `N/A` |
| `Product Image` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Product Name` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Product Price` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Product Status` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `shipping date (DateOrders)` | `str` | `0` | `0.00%` | `2015-01-03 00:00:00` | `2018-02-06 22:14:00` |
| `Shipping Mode` | `str` | `0` | `0.00%` | `N/A` | `N/A` |


### File: `DescriptionDataCoSupplyChain.csv`
- **Dimensions**: `52` rows x `2` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `FIELDS` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `DESCRIPTION` | `str` | `0` | `0.00%` | `N/A` | `N/A` |


### File: `tokenized_access_logs.csv`
- **Dimensions**: `469,977` rows x `8` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Product` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Category` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Date` | `str` | `0` | `0.00%` | `2017-09-01 06:00:00` | `2018-01-31 23:58:00` |
| `Month` | `str` | `0` | `0.00%` | `0001-01-01 00:00:00` | `0001-12-01 00:00:00` |
| `Hour` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Department` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `ip` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `url` | `str` | `0` | `0.00%` | `N/A` | `N/A` |


---

## Dataset: ROSSMANN

**Total Files**: 4

### File: `sample_submission.csv`
- **Dimensions**: `41,088` rows x `2` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Sales` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |


### File: `store.csv`
- **Dimensions**: `1,115` rows x `10` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Store` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `StoreType` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `Assortment` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `CompetitionDistance` | `float64` | `3` | `0.27%` | `N/A` | `N/A` |
| `CompetitionOpenSinceMonth` | `float64` | `354` | `31.75%` | `1970-01-01 00:00:00.000000001` | `1970-01-01 00:00:00.000000012` |
| `CompetitionOpenSinceYear` | `float64` | `354` | `31.75%` | `1970-01-01 00:00:00.000001900` | `1970-01-01 00:00:00.000002015` |
| `Promo2` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Promo2SinceWeek` | `float64` | `544` | `48.79%` | `N/A` | `N/A` |
| `Promo2SinceYear` | `float64` | `544` | `48.79%` | `1970-01-01 00:00:00.000002009` | `1970-01-01 00:00:00.000002015` |
| `PromoInterval` | `str` | `544` | `48.79%` | `N/A` | `N/A` |


### File: `test.csv`
- **Dimensions**: `41,088` rows x `8` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Id` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Store` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `DayOfWeek` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Date` | `str` | `0` | `0.00%` | `2015-08-01 00:00:00` | `2015-09-17 00:00:00` |
| `Open` | `float64` | `11` | `0.03%` | `N/A` | `N/A` |
| `Promo` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `StateHoliday` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `SchoolHoliday` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |


### File: `train.csv`
- **Dimensions**: `1,017,209` rows x `9` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Store` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `DayOfWeek` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Date` | `str` | `0` | `0.00%` | `2013-01-01 00:00:00` | `2015-07-31 00:00:00` |
| `Sales` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Customers` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Open` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Promo` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `StateHoliday` | `str` | `0` | `0.00%` | `N/A` | `N/A` |
| `SchoolHoliday` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |


---

## Dataset: M5

**Total Files**: 1

### File: `Walmart.csv`
- **Dimensions**: `6,435` rows x `8` columns
- **Column Schema & Missingness Analysis**:

| Column Name | Data Type | Missing Count | Missing % | Min Date / Value | Max Date / Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Store` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Date` | `str` | `0` | `0.00%` | `2010-01-10 00:00:00` | `2012-12-10 00:00:00` |
| `Weekly_Sales` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Holiday_Flag` | `int64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Temperature` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Fuel_Price` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `CPI` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |
| `Unemployment` | `float64` | `0` | `0.00%` | `N/A` | `N/A` |


---

## Candidate Primary Keys & Join Keys per Dataset

### 1. Olist Brazilian E-Commerce Dataset
- **Primary Keys**:
  - `olist_customers_dataset.csv`: `customer_id` (Primary Key, unique per order instance)
  - `olist_geolocation_dataset.csv`: `geolocation_zip_code_prefix` (Non-unique location index)
  - `olist_order_items_dataset.csv`: Composite (`order_id`, `order_item_id`)
  - `olist_order_payments_dataset.csv`: Composite (`order_id`, `payment_sequential`)
  - `olist_order_reviews_dataset.csv`: `review_id` (Composite: `review_id`, `order_id`)
  - `olist_orders_dataset.csv`: `order_id` (Primary Key)
  - `olist_products_dataset.csv`: `product_id` (Primary Key)
  - `olist_sellers_dataset.csv`: `seller_id` (Primary Key)
  - `product_category_name_translation.csv`: `product_category_name` (Primary Key)
- **Join Keys & Foreign Relationships**:
  - `order_id`: Links `olist_orders_dataset` to `order_items`, `order_payments`, and `order_reviews`
  - `customer_id`: Links `olist_orders_dataset` to `olist_customers_dataset` (`customer_id` connects to `customer_unique_id`)
  - `product_id`: Links `olist_order_items_dataset` to `olist_products_dataset`
  - `seller_id`: Links `olist_order_items_dataset` to `olist_sellers_dataset`
  - `product_category_name`: Links `olist_products_dataset` to `product_category_name_translation`
  - `zip_code_prefix`: Links `customers`/`sellers` to `olist_geolocation_dataset`

### 2. DataCo Smart Supply Chain Dataset
- **Primary Keys**:
  - `DataCoSupplyChainDataset.csv`: `Order Item Id` (Primary Key per row), composite (`Order Id`, `Order Item Id`)
  - `DescriptionDataCoSupplyChain.csv`: `FIELDS` (Primary Key for field metadata)
  - `tokenized_access_logs.csv`: Transactional event log (No strict primary key; composite timestamp + IP + URL)
- **Join Keys & Foreign Relationships**:
  - `Order Id` / `Order Item Id`: Connects order lines to shipment tracking and financial calculation logic
  - `Customer Id` / `Order Customer Id`: Links customers to order transactions
  - `Product Card Id` / `Category Id` / `Department Id`: Connects order lines to product master catalog and organizational hierarchy

### 3. Rossmann Store Sales Dataset
- **Primary Keys**:
  - `store.csv`: `Store` (Primary Key)
  - `train.csv`: Composite (`Store`, `Date`)
  - `test.csv`: `Id` (Primary Key), composite (`Store`, `Date`)
  - `sample_submission.csv`: `Id` (Primary Key)
- **Join Keys & Foreign Relationships**:
  - `Store`: Primary foreign key joining `train.csv` and `test.csv` to `store.csv` metadata
  - `Date`: Time join key for joining seasonal external calendars and promotional schedules

### 4. M5 Forecasting / Walmart Dataset
- **Primary Keys**:
  - `Walmart.csv`: Composite (`Store`, `Date`)
- **Join Keys & Foreign Relationships**:
  - `Store`: Joins store location attributes and regional parameters
  - `Date`: Time-series join key for macroeconomic indicators (CPI, Fuel Price, Unemployment, Temperature)
