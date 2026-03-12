import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------------
# THEME
# -------------------------

st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:700;
text-align:center;
padding:20px;
margin-bottom:25px;
border:3px solid #00C9A7;
border-radius:12px;
background:linear-gradient(90deg,#141E30,#243B55);
color:white;
}

.kpi-card{
background:#111;
padding:20px;
border-radius:10px;
border:2px solid #00C9A7;
text-align:center;
}

.kpi-title{
font-size:18px;
color:#bbb;
}

.kpi-value{
font-size:32px;
font-weight:bold;
color:#00C9A7;
}

.section{
font-size:24px;
font-weight:600;
margin-top:30px;
border-left:5px solid #00C9A7;
padding-left:10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Amazon Product Intelligence Dashboard</div>', unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------

df = pd.read_csv("amazon_final_cleaned.csv")

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")
df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")

# -------------------------
# FILTERS
# -------------------------

st.sidebar.title("Filters")

category = st.sidebar.multiselect(
"Category",
df["main_category"].unique(),
default=df["main_category"].unique()
)

filtered = df[df["main_category"].isin(category)]

# -------------------------
# KPI
# -------------------------

c1,c2,c3,c4 = st.columns(4)

c1.markdown(f"""
<div class="kpi-card">
<div class="kpi-title">Total Products</div>
<div class="kpi-value">{len(filtered)}</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="kpi-card">
<div class="kpi-title">Average Rating</div>
<div class="kpi-value">{round(filtered["rating"].mean(),2)}</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="kpi-card">
<div class="kpi-title">Average Discount</div>
<div class="kpi-value">{round(filtered["discount_percentage"].mean(),1)}%</div>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="kpi-card">
<div class="kpi-title">Total Reviews</div>
<div class="kpi-value">{int(filtered["rating_count"].sum())}</div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# LINE + AREA
# -------------------------

st.markdown('<div class="section">Sales & Rating Trends</div>', unsafe_allow_html=True)

col1,col2 = st.columns(2)

line = filtered.groupby("main_category")["rating"].mean().reset_index()

fig_line = px.line(line,x="main_category",y="rating",markers=True)

col1.plotly_chart(fig_line,use_container_width=True)

area = filtered.groupby("main_category")["rating_count"].sum().reset_index()

fig_area = px.area(area,x="main_category",y="rating_count")

col2.plotly_chart(fig_area,use_container_width=True)

# -------------------------
# PIE + DONUT
# -------------------------

st.markdown('<div class="section">Category Insights</div>', unsafe_allow_html=True)

col3,col4 = st.columns(2)

fig_pie = px.pie(filtered,names="main_category",title="Product Distribution")

col3.plotly_chart(fig_pie,use_container_width=True)

donut = filtered.groupby("main_category")["discount_percentage"].mean().reset_index()

fig_donut = px.pie(donut,names="main_category",values="discount_percentage",hole=0.5,title="Average Discount")

col4.plotly_chart(fig_donut,use_container_width=True)

# -------------------------
# STACKED + CLUSTER
# -------------------------

st.markdown('<div class="section">Pricing Insights</div>', unsafe_allow_html=True)

col5,col6 = st.columns(2)

stack = filtered.groupby(["main_category","rating"]).size().reset_index(name="count")

fig_stack = px.bar(stack,x="main_category",y="count",color="rating",title="Rating Distribution")

col5.plotly_chart(fig_stack,use_container_width=True)

cluster = filtered.groupby("main_category")[["discount_percentage","discounted_price"]].mean().reset_index()

fig_cluster = px.bar(cluster,x="main_category",y=["discount_percentage","discounted_price"],barmode="group",title="Price vs Discount")

col6.plotly_chart(fig_cluster,use_container_width=True)

# -------------------------
# SCATTER
# -------------------------

st.markdown('<div class="section">Price vs Rating</div>', unsafe_allow_html=True)

fig_scatter = px.scatter(
filtered,
x="discounted_price",
y="rating",
size="rating_count",
color="main_category"
)

st.plotly_chart(fig_scatter,use_container_width=True)

# -------------------------
# TOP PRODUCTS
# -------------------------

st.markdown('<div class="section">Top 10 Products</div>', unsafe_allow_html=True)

top10 = filtered.sort_values("rating_count",ascending=False).head(10)

fig_top = px.bar(
top10,
x="rating_count",
y="product_name",
orientation="h"
)

st.plotly_chart(fig_top,use_container_width=True)

# -------------------------
# MATRIX
# -------------------------

st.markdown('<div class="section">Product Matrix</div>', unsafe_allow_html=True)

matrix = filtered.pivot_table(
values="rating_count",
index="main_category",
columns="rating",
aggfunc="sum",
fill_value=0
)

st.dataframe(matrix)

# -------------------------
# MAP
# -------------------------

map_data = pd.DataFrame({
"lat":[28.61,19.07,18.52,12.97],
"lon":[77.20,72.87,73.85,77.59]
})

st.map(map_data)

# -------------------------
# WORD CLOUD
# -------------------------

st.markdown('<div class="section">Review Word Cloud</div>', unsafe_allow_html=True)

text = " ".join(filtered["review_content"].dropna())

wc = WordCloud(width=800,height=400).generate(text)

fig,ax = plt.subplots()

ax.imshow(wc)

ax.axis("off")

st.pyplot(fig)

# -------------------------
# AI RECOMMENDATION
# -------------------------

st.markdown('<div class="section">AI Product Recommendation</div>', unsafe_allow_html=True)

filtered["features"] = filtered["product_name"] + " " + filtered["main_category"]

tfidf = TfidfVectorizer(stop_words="english")

matrix = tfidf.fit_transform(filtered["features"])

similarity = cosine_similarity(matrix)

product_list = filtered["product_name"].unique()

selected = st.selectbox("Select Product",product_list)

idx = filtered[filtered["product_name"]==selected].index[0]

scores = list(enumerate(similarity[idx]))

scores = sorted(scores,key=lambda x:x[1],reverse=True)

recommended = []

for i in scores:

    name = filtered.iloc[i[0]]["product_name"]

    if name!=selected and name not in recommended:

        recommended.append(name)

    if len(recommended)==5:

        break

for r in recommended:

    st.write("•",r)

# -------------------------
# SENTIMENT
# -------------------------

st.markdown('<div class="section">Customer Sentiment</div>', unsafe_allow_html=True)

def sentiment(text):

    score = TextBlob(str(text)).sentiment.polarity

    if score>0:

        return "Positive"

    elif score<0:

        return "Negative"

    else:

        return "Neutral"

reviews = filtered["review_content"].dropna().sample(200)

sentiments = reviews.apply(sentiment)

sent_df = sentiments.value_counts().reset_index()

sent_df.columns=["Sentiment","Count"]

fig_sent = px.pie(sent_df,names="Sentiment",values="Count")

st.plotly_chart(fig_sent)

# -------------------------
# NOTES
# -------------------------

st.markdown('<div class="section">Dashboard Insights</div>', unsafe_allow_html=True)

st.write("""
• Dashboard shows product rating and discount insights.

• AI recommendation suggests similar products.

• Sentiment analysis shows customer satisfaction.

• Filters allow dynamic exploration of categories.
""")