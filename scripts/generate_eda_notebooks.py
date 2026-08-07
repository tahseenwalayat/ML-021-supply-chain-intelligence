import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("notebooks", exist_ok=True)


def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernel_spec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}


print("Generating EDA Notebooks...")

# ==============================================================================
# NOTEBOOK 1: 01_eda_sales_overview.ipynb
# ==============================================================================
nb1_cells = [
    md_cell("# EDA 01: Enterprise Sales Overview & Trend Analysis\n\nThis notebook analyzes total sales revenue, transaction volumes, growth trajectories over time, and sales distributions across raw dataset sources and geographic regions."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)

print("Loading sales_fact.parquet...")
df_sales = pd.read_parquet('data/processed/sales_fact.parquet')
print(f"Total Rows: {len(df_sales):,}, Columns: {df_sales.shape[1]}")
print(f"Date Range: {df_sales['date'].min()} to {df_sales['date'].max()}")
df_sales.head()
"""),
    code_cell("""# Monthly Aggregation of Total Revenue & Transaction Counts
df_sales['year_month'] = df_sales['date'].dt.to_period('M')
monthly_sales = df_sales.groupby('year_month').agg(
    total_revenue=('total_sales', 'sum'),
    transaction_count=('sales_id', 'count'),
    avg_ticket=('total_sales', 'mean')
).reset_index()
monthly_sales['year_month_str'] = monthly_sales['year_month'].astype(str)

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

ax1.plot(monthly_sales['year_month_str'], monthly_sales['total_revenue'] / 1e6, 'b-o', label='Monthly Revenue ($M)', linewidth=2)
ax2.plot(monthly_sales['year_month_str'], monthly_sales['transaction_count'], 'g--s', label='Transaction Count', linewidth=2)

ax1.set_xlabel('Year-Month')
ax1.set_ylabel('Total Revenue ($ Millions)', color='b')
ax2.set_ylabel('Transaction Count', color='g')
plt.title('Enterprise Monthly Sales Revenue and Transaction Volume (2010 - 2018)')
ax1.tick_params(axis='x', rotation=45)
fig.tight_layout()
plt.show()
"""),
    code_cell("""# Revenue & Volume Breakdown by Dataset Source
source_summary = df_sales.groupby('dataset_source').agg(
    total_revenue=('total_sales', 'sum'),
    total_volume=('quantity', 'sum'),
    transaction_count=('sales_id', 'count'),
    mean_sale=('total_sales', 'mean'),
    median_sale=('total_sales', 'median')
).reset_index()

source_summary['revenue_share_pct'] = (source_summary['total_revenue'] / source_summary['total_revenue'].sum()) * 100
print(source_summary)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=source_summary, x='dataset_source', y='total_revenue', ax=ax1, palette='viridis')
ax1.set_title('Total Gross Revenue by Dataset Source ($)')
ax1.set_ylabel('Revenue ($)')

sns.barplot(data=source_summary, x='dataset_source', y='transaction_count', ax=ax2, palette='magma')
ax2.set_title('Transaction Volume by Dataset Source')
ax2.set_ylabel('Transaction Count')
plt.tight_layout()
plt.show()
"""),
    code_cell("""# Top 10 Geographic Regions by Revenue
region_summary = df_sales.groupby('region').agg(
    total_revenue=('total_sales', 'sum'),
    transaction_count=('sales_id', 'count')
).reset_index().sort_values(by='total_revenue', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=region_summary.head(10), x='total_revenue', y='region', palette='Spectral')
plt.title('Top 10 Regions by Total Revenue ($)')
plt.xlabel('Total Revenue ($)')
plt.ylabel('Region')
plt.tight_layout()
plt.show()
"""),
    code_cell("""# Statistical Summary of Sales Metrics
sales_stats = df_sales[['total_sales', 'quantity', 'unit_price', 'shipping_cost', 'profit']].describe().T
sales_stats['skewness'] = df_sales[['total_sales', 'quantity', 'unit_price', 'shipping_cost', 'profit']].skew()
print("Sales Fact Descriptive Statistics:")
print(sales_stats)
""")
]

with open("notebooks/01_eda_sales_overview.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_nb(nb1_cells), f, indent=2)


# ==============================================================================
# NOTEBOOK 2: 02_eda_seasonality_analysis.ipynb
# ==============================================================================
nb2_cells = [
    md_cell("# EDA 02: Seasonality, Day-of-Week, and Holiday Effect Analysis\n\nThis notebook evaluates seasonal demand cycles across days of the week, months of the year, and public holidays using `calendar_dim`."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)

df_sales = pd.read_parquet('data/processed/sales_fact.parquet')
df_cal = pd.read_parquet('data/processed/calendar_dim.parquet')

# Merge sales_fact with calendar_dim on date
df_merged = df_sales.merge(df_cal, on='date', how='inner')
print(f"Merged Dataset Rows: {len(df_merged):,}")
df_merged.head()
"""),
    code_cell("""# Day-of-Week Seasonality (0=Monday, 6=Sunday)
dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df_merged['dow_name'] = df_merged['day_of_week'].map(dow_names)

dow_summary = df_merged.groupby(['day_of_week', 'dow_name']).agg(
    total_revenue=('total_sales', 'sum'),
    avg_daily_revenue=('total_sales', 'mean'),
    transaction_count=('sales_id', 'count')
).reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=dow_summary, x='dow_name', y='avg_daily_revenue', palette='Blues_d')
plt.title('Average Revenue per Transaction by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Mean Transaction Revenue ($)')
plt.tight_layout()
plt.show()
"""),
    code_cell("""# Monthly Seasonality (Month 1 to 12)
month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
df_merged['month_name'] = df_merged['month'].map(month_names)

monthly_agg = df_merged.groupby(['month', 'month_name']).agg(
    total_revenue=('total_sales', 'sum'),
    mean_revenue=('total_sales', 'mean'),
    transaction_count=('sales_id', 'count')
).reset_index()

plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly_agg, x='month_name', y='total_revenue', marker='o', color='crimson', linewidth=2.5)
plt.title('Total Revenue Seasonality by Month of Year')
plt.xlabel('Month')
plt.ylabel('Total Revenue ($)')
plt.tight_layout()
plt.show()
"""),
    code_cell("""# Weekend vs Weekday Sales Analysis
weekend_summary = df_merged.groupby('is_weekend').agg(
    total_revenue=('total_sales', 'sum'),
    mean_revenue=('total_sales', 'mean'),
    transaction_count=('sales_id', 'count')
).reset_index()

weekend_summary['type'] = np.where(weekend_summary['is_weekend'], 'Weekend', 'Weekday')
print(weekend_summary)

plt.figure(figsize=(8, 5))
sns.barplot(data=weekend_summary, x='type', y='mean_revenue', palette='Set2')
plt.title('Average Order Revenue: Weekday vs Weekend')
plt.ylabel('Mean Sales ($)')
plt.show()
"""),
    code_cell("""# Holiday Effect Analysis (Holiday vs Non-Holiday)
holiday_summary = df_merged.groupby('is_holiday').agg(
    total_revenue=('total_sales', 'sum'),
    mean_sales=('total_sales', 'mean'),
    median_sales=('total_sales', 'median'),
    transaction_count=('sales_id', 'count')
).reset_index()

holiday_summary['type'] = np.where(holiday_summary['is_holiday'], 'Holiday Day', 'Regular Day')
print(holiday_summary)

reg_mean = holiday_summary.loc[~holiday_summary['is_holiday'], 'mean_sales'].values[0]
hol_mean = holiday_summary.loc[holiday_summary['is_holiday'], 'mean_sales'].values[0]
lift_pct = ((hol_mean - reg_mean) / reg_mean) * 100
print(f"Holiday Revenue Lift: {lift_pct:.2f}%")
""")
]

with open("notebooks/02_eda_seasonality_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_nb(nb2_cells), f, indent=2)


# ==============================================================================
# NOTEBOOK 3: 03_eda_promotion_lift.ipynb
# ==============================================================================
nb3_cells = [
    md_cell("# EDA 03: Promotion Lift & Discount Effectiveness Analysis\n\nThis notebook evaluates sales lift, order volume changes, and discount impact under active promotions across datasets."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)

df_sales = pd.read_parquet('data/processed/sales_fact.parquet')
df_promo = pd.read_parquet('data/processed/promotion_dim.parquet')

# Drop dataset_source from promo to prevent suffix collision
df_merged = df_sales.merge(df_promo.drop(columns=['dataset_source'], errors='ignore'), on='promotion_id', how='left')
df_merged['is_promo_active'] = df_merged['promotion_id'] != 'promo_none'
print(f"Total Rows: {len(df_merged):,}")
print(df_merged['is_promo_active'].value_counts(normalize=True) * 100)
"""),
    code_cell("""# Overall Promotion vs Non-Promotion Sales Metrics
promo_summary = df_merged.groupby('is_promo_active').agg(
    total_revenue=('total_sales', 'sum'),
    mean_sales=('total_sales', 'mean'),
    median_sales=('total_sales', 'median'),
    mean_quantity=('quantity', 'mean'),
    mean_discount=('discount_amount', 'mean'),
    transaction_count=('sales_id', 'count')
).reset_index()

promo_summary['status'] = np.where(promo_summary['is_promo_active'], 'Promotional', 'Baseline (No Promo)')
print(promo_summary)

no_promo_val = promo_summary.loc[~promo_summary['is_promo_active'], 'mean_sales'].values[0]
promo_val = promo_summary.loc[promo_summary['is_promo_active'], 'mean_sales'].values[0]
overall_lift = ((promo_val - no_promo_val) / no_promo_val) * 100
print(f"Overall Sales Lift from Promotions: {overall_lift:.2f}%")
"""),
    code_cell("""# Promotion Lift Breakdown by Dataset Source
source_promo = df_merged.groupby(['dataset_source', 'is_promo_active']).agg(
    mean_sales=('total_sales', 'mean'),
    total_revenue=('total_sales', 'sum'),
    transaction_count=('sales_id', 'count')
).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=source_promo, x='dataset_source', y='mean_sales', hue='is_promo_active', palette='PuBuGn')
plt.title('Average Sales Revenue by Dataset Source: Promotional vs Baseline')
plt.xlabel('Dataset Source')
plt.ylabel('Mean Transaction Sales ($)')
plt.legend(title='Is Promo Active', labels=['No Promo', 'Promo'])
plt.tight_layout()
plt.show()
"""),
    code_cell("""# Detailed Lift by Promotion Type
promo_type_summary = df_merged.groupby(['promotion_id', 'promo_name', 'discount_type']).agg(
    mean_sales=('total_sales', 'mean'),
    total_revenue=('total_sales', 'sum'),
    mean_discount=('discount_amount', 'mean'),
    transaction_count=('sales_id', 'count')
).reset_index()

print(promo_type_summary)

plt.figure(figsize=(10, 5))
sns.barplot(data=promo_type_summary, x='promo_name', y='mean_sales', palette='YlOrRd')
plt.title('Mean Sales Revenue per Promotion Campaign')
plt.xlabel('Promotion Name')
plt.ylabel('Mean Sales ($)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
""")
]

with open("notebooks/03_eda_promotion_lift.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_nb(nb3_cells), f, indent=2)


# ==============================================================================
# NOTEBOOK 4: 04_eda_delivery_performance.ipynb
# ==============================================================================
nb4_cells = [
    md_cell("# EDA 04: Delivery Performance, Freight Costs & Supplier Efficiency\n\nThis notebook evaluates freight costs, supplier lead times, reliability scores, and net profit margins across operating regions."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)

df_sales = pd.read_parquet('data/processed/sales_fact.parquet')
df_supp = pd.read_parquet('data/processed/supplier_dim.parquet')
df_wh = pd.read_parquet('data/processed/warehouse_dim.parquet')

print(f"Sales Records: {len(df_sales):,}, Suppliers: {len(df_supp):,}, Warehouses: {len(df_wh):,}")
"""),
    code_cell("""# Shipping Cost Distribution & Freight Ratio
df_sales['freight_ratio'] = np.where(df_sales['total_sales'] > 0, df_sales['shipping_cost'] / df_sales['total_sales'], 0.0)

freight_stats = df_sales[['shipping_cost', 'freight_ratio', 'profit']].describe().T
print("Shipping Cost and Freight Ratio Statistics:")
print(freight_stats)

plt.figure(figsize=(10, 5))
sns.histplot(df_sales[df_sales['shipping_cost'] > 0]['shipping_cost'], bins=50, kde=True, color='teal')
plt.title('Distribution of Shipping Freight Costs ($)')
plt.xlabel('Shipping Cost ($)')
plt.ylabel('Frequency')
plt.show()
"""),
    code_cell("""# Regional Profit Margin & Shipping Cost Analysis
region_delivery = df_sales.groupby('region').agg(
    total_revenue=('total_sales', 'sum'),
    total_freight=('shipping_cost', 'sum'),
    total_profit=('profit', 'sum'),
    avg_freight=('shipping_cost', 'mean'),
    avg_profit=('profit', 'mean'),
    order_count=('sales_id', 'count')
).reset_index().sort_values(by='total_revenue', ascending=False)

region_delivery['profit_margin_pct'] = np.where(region_delivery['total_revenue'] > 0, (region_delivery['total_profit'] / region_delivery['total_revenue']) * 100, 0.0)
print(region_delivery.head(10))

plt.figure(figsize=(12, 6))
sns.barplot(data=region_delivery.head(10), x='avg_freight', y='region', palette='mako')
plt.title('Top 10 Regions by Average Shipping Freight Cost ($)')
plt.xlabel('Average Freight Cost ($)')
plt.ylabel('Region')
plt.tight_layout()
plt.show()
"""),
    code_cell("""# Supplier Lead Time & Reliability Analysis
supp_merged = df_sales.merge(df_supp.drop(columns=['dataset_source', 'region', 'city', 'country'], errors='ignore'), on='supplier_id', how='left')

supp_perf = supp_merged.groupby(['supplier_id', 'supplier_name', 'lead_time_days', 'reliability_score']).agg(
    order_count=('sales_id', 'count'),
    total_revenue=('total_sales', 'sum'),
    avg_profit=('profit', 'mean')
).reset_index().sort_values(by='order_count', ascending=False)

print("Top Suppliers by Volume & Performance:")
print(supp_perf.head(10))
""")
]

with open("notebooks/04_eda_delivery_performance.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_nb(nb4_cells), f, indent=2)


# ==============================================================================
# NOTEBOOK 5: 05_eda_regional_demand_patterns.ipynb
# ==============================================================================
nb5_cells = [
    md_cell("# EDA 05: Regional Demand Patterns, Category Shares & Weather Correlation\n\nThis notebook evaluates product category demand variations, demand volatility (Coefficient of Variation), and correlation with synthetic weather variables."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)

df_sales = pd.read_parquet('data/processed/sales_fact.parquet')
df_prod = pd.read_parquet('data/processed/product_dim.parquet')
df_weather = pd.read_parquet('data/processed/weather_dim.parquet')

df_merged = df_sales.merge(df_prod[['product_id', 'category', 'sub_category']], on='product_id', how='left')
print(f"Sales with Category Info: {len(df_merged):,}")
"""),
    code_cell("""# Regional Category Sales Heatmap
cat_region_pivot = pd.pivot_table(
    df_merged,
    values='total_sales',
    index='category',
    columns='dataset_source',
    aggfunc='sum',
    fill_value=0
)

plt.figure(figsize=(10, 8))
sns.heatmap(cat_region_pivot / 1e6, annot=True, fmt='.2f', cmap='YlGnBu')
plt.title('Category Revenue Breakdown by Dataset Source ($ Millions)')
plt.xlabel('Dataset Source')
plt.ylabel('Product Category')
plt.tight_layout()
plt.show()
"""),
    code_cell("""# Demand Volatility Analysis (Coefficient of Variation = std / mean)
daily_reg_sales = df_sales.groupby(['region', 'date'])['total_sales'].sum().reset_index()

cv_summary = daily_reg_sales.groupby('region')['total_sales'].agg(
    mean_sales='mean',
    std_sales='std',
    days_recorded='count'
).reset_index()

cv_summary['cv'] = cv_summary['std_sales'] / cv_summary['mean_sales']
cv_summary['volatility_tier'] = pd.qcut(cv_summary['cv'], q=3, labels=['Low Volatility', 'Medium Volatility', 'High Volatility'])

print(cv_summary.sort_values(by='cv', ascending=False).head(10))

plt.figure(figsize=(10, 5))
sns.barplot(data=cv_summary.sort_values(by='cv', ascending=False).head(10), x='cv', y='region', palette='OrRd_r')
plt.title('Top 10 Regions by Demand Volatility (Coefficient of Variation)')
plt.xlabel('CV (std / mean)')
plt.ylabel('Region')
plt.show()
"""),
    code_cell("""# Weather Parameter Correlation Analysis
df_sales['date_clean'] = df_sales['date'].dt.date
df_weather['date_clean'] = df_weather['date'].dt.date

daily_sales_reg = df_sales.groupby(['region', 'date_clean'])['total_sales'].sum().reset_index()
weather_merged = daily_sales_reg.merge(df_weather, on=['region', 'date_clean'], how='inner')

corr_matrix = weather_merged[['total_sales', 'temperature_c', 'precipitation_mm', 'humidity_pct']].corr()
print("Correlation Matrix (Daily Sales vs Synthetic Weather):")
print(corr_matrix)

plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation: Daily Sales vs Synthetic Weather Parameters')
plt.show()
""")
]

with open("notebooks/05_eda_regional_demand_patterns.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_nb(nb5_cells), f, indent=2)


# ==============================================================================
# NOTEBOOK 6: 06_eda_data_quality.ipynb
# ==============================================================================
nb6_cells = [
    md_cell("# EDA 06: Enterprise Data Quality, Missingness & Outlier Audit\n\nThis notebook performs a comprehensive data quality audit across all 8 processed tables, evaluating completeness, primary key uniqueness, data type integrity, and extreme value distributions."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)

tables = ['sales_fact', 'product_dim', 'warehouse_dim', 'supplier_dim', 'promotion_dim', 'calendar_dim', 'weather_dim', 'event_dim']
table_dfs = {t: pd.read_parquet(f'data/processed/{t}.parquet') for t in tables}

print("Loaded all 8 processed tables successfully.")
"""),
    code_cell("""# Table Completeness Audit (Null Percentage per Column)
null_audit = []
for name, df in table_dfs.items():
    total_rows = len(df)
    null_cols = df.isnull().sum()
    for col, null_cnt in null_cols.items():
        null_pct = (null_cnt / total_rows) * 100
        null_audit.append({
            'table_name': name,
            'column_name': col,
            'dtype': str(df[col].dtype),
            'total_rows': total_rows,
            'null_count': null_cnt,
            'null_pct': round(null_pct, 4)
        })

df_null_audit = pd.DataFrame(null_audit)
print("Columns with any missing values:")
print(df_null_audit[df_null_audit['null_count'] > 0])
if (df_null_audit['null_count'] == 0).all():
    print("ALL 8 TABLES HAVE 0% NULL VALUES ACROSS ALL COLUMNS!")
"""),
    code_cell("""# Primary Key Uniqueness Audit
pk_map = {
    'sales_fact': 'sales_id',
    'product_dim': 'product_id',
    'warehouse_dim': 'warehouse_id',
    'supplier_dim': 'supplier_id',
    'promotion_dim': 'promotion_id',
    'calendar_dim': 'date',
    'weather_dim': 'weather_id',
    'event_dim': 'event_id'
}

pk_audit = []
for name, df in table_dfs.items():
    pk_col = pk_map[name]
    total_count = len(df)
    unique_count = df[pk_col].nunique()
    dupe_count = total_count - unique_count
    pk_audit.append({
        'table_name': name,
        'primary_key': pk_col,
        'total_rows': total_count,
        'unique_keys': unique_count,
        'duplicate_keys': dupe_count,
        'is_unique': dupe_count == 0
    })

df_pk_audit = pd.DataFrame(pk_audit)
print("Primary Key Audit Summary:")
print(df_pk_audit)
"""),
    code_cell("""# Outlier & Skewness Audit for Sales Fact Metrics
df_sales = table_dfs['sales_fact']
numeric_cols = ['total_sales', 'quantity', 'unit_price', 'shipping_cost', 'profit']

num_audit = []
for col in numeric_cols:
    s = df_sales[col]
    num_audit.append({
        'metric': col,
        'min': s.min(),
        'p01': s.quantile(0.01),
        'p50 (median)': s.median(),
        'mean': s.mean(),
        'p99': s.quantile(0.99),
        'max': s.max(),
        'std': s.std(),
        'skewness': s.skew()
    })

df_num_audit = pd.DataFrame(num_audit)
print("Numerical Distribution & Winsorization Audit:")
print(df_num_audit)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_pk_audit, x='table_name', y='total_rows', palette='crest')
plt.title('Total Record Count per Unified Table')
plt.ylabel('Row Count (Log Scale)')
plt.yscale('log')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
""")
]

with open("notebooks/06_eda_data_quality.ipynb", "w", encoding="utf-8") as f:
    json.dump(create_nb(nb6_cells), f, indent=2)

print("All 6 EDA notebooks generated successfully.")
