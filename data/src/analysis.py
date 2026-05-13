import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# =========================
# CREATE FOLDER FOR IMAGES
# =========================
os.makedirs("images", exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/sales_data.csv")

print("\nFirst 5 rows:\n", df.head())

# =========================
# CLEANING
# =========================
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# =========================
# FEATURE ENGINEERING
# =========================
df["Total_Sales"] = df["Sales"] * df["Quantity"]

# =========================
# ANALYSIS
# =========================

product_sales = df.groupby("Product")["Total_Sales"].sum().sort_values()
region_sales = df.groupby("Region")["Total_Sales"].sum()
category_sales = df.groupby("Category")["Total_Sales"].sum()

print("\nProduct Sales:\n", product_sales)
print("\nRegion Sales:\n", region_sales)

# =========================
# 1. PRODUCT SALES
# =========================
plt.figure(figsize=(8,5))
product_sales.plot(kind="barh", color="skyblue")
plt.title("Product Wise Sales")
plt.xlabel("Total Sales")
plt.tight_layout()
plt.savefig("images/product_sales.png")
plt.show()

# =========================
# 2. REGION SALES
# =========================
plt.figure(figsize=(6,6))
region_sales.plot(kind="pie", autopct="%1.1f%%")
plt.title("Region Wise Sales")
plt.ylabel("")
plt.savefig("images/region_sales.png")
plt.show()

# =========================
# 3. CATEGORY SALES
# =========================
plt.figure(figsize=(6,4))
category_sales.plot(kind="bar", color="orange")
plt.title("Category Sales")
plt.ylabel("Sales")
plt.savefig("images/category_sales.png")
plt.show()

# =========================
# 4. SALES DISTRIBUTION
# =========================
plt.figure(figsize=(7,5))
sns.histplot(df["Total_Sales"], bins=10, kde=True, color="green")
plt.title("Sales Distribution")
plt.savefig("images/sales_distribution.png")
plt.show()

# =========================
# 5. HEATMAP
# =========================
pivot = df.pivot_table(values="Total_Sales", index="Product", columns="Region", aggfunc="sum")

plt.figure(figsize=(8,5))
sns.heatmap(pivot, annot=True, cmap="Blues")
plt.title("Product vs Region Heatmap")
plt.savefig("images/heatmap.png")
plt.show()

print("\nAnalysis Completed Successfully 🚀")
