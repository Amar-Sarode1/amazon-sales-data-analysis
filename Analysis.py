import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from textblob import TextBlob


# ---------------------------------------
# 1. LOAD DATASET
# ---------------------------------------

try:
    df = pd.read_csv("amazon.csv")
    print("Dataset loaded successfully")
except FileNotFoundError:
    print("Dataset file not found")
    exit()


# ---------------------------------------
# 2. BASIC DATA CLEANING
# ---------------------------------------

price_columns = ["discounted_price", "actual_price"]

for col in price_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df[col] = pd.to_numeric(df[col], errors="coerce")


# clean discount percentage
df["discount_percentage"] = (
    df["discount_percentage"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["discount_percentage"] = pd.to_numeric(
    df["discount_percentage"], errors="coerce"
)


# clean rating column
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating"] = df["rating"].fillna(df["rating"].median())


# clean rating count
df["rating_count"] = (
    df["rating_count"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
df["rating_count"] = df["rating_count"].fillna(0)


# extract main category
df["main_category"] = df["category"].astype(str).str.split("|").str[0]


# ---------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------

# discount amount
df["discount_amount"] = df["actual_price"] - df["discounted_price"]

# popularity score
df["popularity_score"] = df["rating"] * np.log1p(df["rating_count"])

# value for money score
df["vfm_score"] = (df["rating"] * 0.7) + ((df["discount_percentage"] / 20) * 0.3)


# create price tiers
df["price_tier"] = pd.qcut(
    df["discounted_price"],
    3,
    labels=["Budget", "Value", "Premium"]
)


# ---------------------------------------
# 4. OUTLIER DETECTION
# ---------------------------------------

Q1 = df["discounted_price"].quantile(0.25)
Q3 = df["discounted_price"].quantile(0.75)

IQR = Q3 - Q1

outliers = df[
    (df["discounted_price"] < (Q1 - 1.5 * IQR)) |
    (df["discounted_price"] > (Q3 + 1.5 * IQR))
]

print("Number of price outliers:", len(outliers))


# ---------------------------------------
# 5. PRODUCT RECOMMENDATION
# ---------------------------------------

recommended_products = (
    df.sort_values(["vfm_score", "rating"], ascending=False)
    .head(10)
)

print("\nTop Recommended Products")
print(
    recommended_products[
        ["product_name", "main_category", "rating", "discount_percentage"]
    ]
)


# ---------------------------------------
# 6. SENTIMENT ANALYSIS
# ---------------------------------------

def sentiment_label(text):

    analysis = TextBlob(str(text))

    if analysis.sentiment.polarity > 0:
        return "Positive"

    elif analysis.sentiment.polarity < 0:
        return "Negative"

    else:
        return "Neutral"


if "review_content" in df.columns:

    df["sentiment"] = df["review_content"].apply(sentiment_label)

    print("\nSentiment Distribution")
    print(df["sentiment"].value_counts())


# ---------------------------------------
# 7. VISUALIZATION
# ---------------------------------------

sns.set_theme(style="darkgrid")


# Category distribution
plt.figure(figsize=(10,6))
sns.countplot(
    data=df,
    y="main_category",
    order=df["main_category"].value_counts().index,
    palette="viridis"
)

plt.title("Product Distribution by Category")
plt.savefig("category_distribution.png")
plt.close()


# Price vs rating
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="discounted_price",
    y="rating",
    hue="price_tier"
)

plt.title("Price vs Rating")
plt.savefig("price_vs_rating.png")
plt.close()


# VFM score distribution
plt.figure(figsize=(8,6))
sns.histplot(df["vfm_score"], kde=True)

plt.title("Value for Money Score Distribution")
plt.savefig("vfm_distribution.png")
plt.close()


# correlation heatmap
plt.figure(figsize=(8,6))

corr = df[
    [
        "actual_price",
        "discounted_price",
        "discount_percentage",
        "rating",
        "rating_count"
    ]
].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.close()


# rating trend across price bins
df["price_bin"] = pd.cut(df["discounted_price"], bins=10)

trend = df.groupby("price_bin")["rating"].mean()

plt.figure(figsize=(10,6))
trend.plot(marker="o")

plt.title("Average Rating Across Price Segments")
plt.xlabel("Price Range")
plt.ylabel("Average Rating")

plt.savefig("rating_trend.png")
plt.close()


# ---------------------------------------
# 8. INTERACTIVE PLOTLY CHART
# ---------------------------------------

fig = px.scatter(
    df,
    x="discounted_price",
    y="rating",
    color="main_category",
    size="rating_count",
    title="Interactive Price vs Rating Analysis"
)

fig.write_html("interactive_dashboard.html")


# ---------------------------------------
# 9. EXPORT CLEAN DATA
# ---------------------------------------

df.to_csv("amazon_cleaned_data.csv", index=False)

print("\nAnalysis completed")
print("Clean dataset saved as amazon_cleaned_data.csv")
print("Charts and dashboard generated successfully")