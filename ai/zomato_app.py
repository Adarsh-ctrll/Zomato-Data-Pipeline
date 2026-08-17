import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector

from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Zomato AI Analytics",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHAT_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

NEW_REVIEWS = 500
TOP_K = 5

CACHE_FILE = "review_embeddings.parquet"

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():

    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = get_embedding_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    .card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 20px;
    }

    .small-text {
        color: #6b7280;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🍴 Zomato AI")

    st.caption("Analytics & Customer Intelligence")

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Ask Your Data",
            "💬 Customer Reviews",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.caption(
        "Powered by Snowflake + dbt + Groq"
    )


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================

@st.cache_resource
def get_connection():

    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema="MARTS",
        role="DBT_ROLE"
    )


# ============================================================
# TEXT-TO-SQL CONFIG
# ============================================================

FORBIDDEN_WORDS = [
    "drop",
    "delete",
    "truncate",
    "alter",
    "update",
    "insert",
    "create",
    "replace",
    "grant",
    "revoke"
]


EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method"
]


SCHEMA = """

Tables available in Snowflake.

Use bare table names only.

FCT_ORDERS(
    order_id,
    order_date,
    customer_id,
    restaurant_id,
    city,
    cuisine,
    payment_method,
    order_status,
    is_delivered,
    sales_amount,
    discount,
    delivery_fee,
    gst,
    customer_rating,
    delivery_time_min
)

DIM_RESTAURANT(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    rating,
    cost_for_two
)

DIM_CUSTOMER(
    customer_id,
    customer_name,
    age,
    age_segment,
    gender,
    city
)

MART_DAILY_CITY_REVENUE(
    order_date,
    city,
    orders,
    cancel_rate,
    gmv,
    aov
)

MART_RESTAURANT_PERFORMANCE(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    orders,
    revenue,
    avg_customer_rating,
    cancel_rate
)

MART_DELIVERY_SLA(
    city,
    order_hour,
    delivered_orders,
    p50_delivery_min,
    late_rate
)

Important:
- gmv means delivered revenue.
- Prefer MART tables when they fit the question.
"""


SYSTEM_PROMPT = f"""
You are a Snowflake SQL expert.

Write ONE SELECT query that answers the user's question.

Rules:

- SELECT queries only.
- Never modify data.
- Use bare table names.
- Do not use database/schema prefixes.
- Add LIMIT 100 or less unless the question asks for a single total.
- Return JSON only.

Exact format:

{{"sql": "your query here"}}

{SCHEMA}
"""


# ============================================================
# TEXT-TO-SQL FUNCTIONS
# ============================================================

def generate_sql(question):

    response = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    answer = response.choices[0].message.content

    sql = json.loads(answer)["sql"]

    sql = sql.replace("ZOMATO.MARTS.", "")
    sql = sql.replace("ZOMATO.", "")

    return sql.strip().rstrip(";")


def is_safe(sql):

    lowered = sql.lower().strip()

    if not (
        lowered.startswith("select")
        or lowered.startswith("with")
    ):
        return False

    for word in FORBIDDEN_WORDS:

        if word in lowered:
            return False

    return True


def run_query(sql):

    conn = get_connection()

    cursor = conn.cursor()

    try:

        return cursor.execute(sql).fetch_pandas_all()

    finally:

        cursor.close()


# ============================================================
# RAG FUNCTIONS
# ============================================================

def read_reviews_from_snowflake():

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT
            REVIEW_ID,
            CITY,
            RATING,
            COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """

    cursor = conn.cursor()

    try:

        df = cursor.execute(query).fetch_pandas_all()

    finally:

        cursor.close()
        conn.close()

    df.columns = [
        col.lower()
        for col in df.columns
    ]

    return df


def embed(texts):

    return embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()


@st.cache_data
def load_reviews():

    if os.path.exists(CACHE_FILE):

        return pd.read_parquet(
            CACHE_FILE
        )

    df = read_reviews_from_snowflake()

    df["embedding"] = embed(
        df["comment"].fillna("").tolist()
    )

    df.to_parquet(
        CACHE_FILE
    )

    return df


def cosine_similarity(vec_a, vec_b):

    return np.dot(
        vec_a,
        vec_b
    ) / (
        np.linalg.norm(vec_a)
        * np.linalg.norm(vec_b)
    )


def find_similar_reviews(
    question,
    df
):

    question_vector = embed(
        [question]
    )[0]

    scores = []

    for review_vector in df["embedding"]:

        scores.append(
            cosine_similarity(
                question_vector,
                review_vector
            )
        )

    result = df.copy()

    result["score"] = scores

    return result.nlargest(
        TOP_K,
        "score"
    )


def ask_llm(
    question,
    top_reviews
):

    context = ""

    for _, row in top_reviews.iterrows():

        context += (
            f"City: {row['city']}, "
            f"Rating: {row['rating']} stars\n"
            f"Review: {row['comment']}\n\n"
        )

    system_prompt = """
    You are a customer review analyst.

    Answer ONLY using the customer reviews provided.

    Be concise and useful.

    If the provided reviews do not contain
    enough information to answer the question,
    clearly say that the available reviews
    do not provide enough evidence.
    """

    user_prompt = f"""
    Question:
    {question}

    Customer Reviews:
    {context}
    """

    response = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">'
        '🍴 Zomato AI Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore business performance and customer feedback '
        'using natural language.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Data Warehouse",
            "Snowflake"
        )

    with col2:

        st.metric(
            "AI Model",
            "Llama 3.3 70B"
        )

    with col3:

        st.metric(
            "Review Search",
            f"{NEW_REVIEWS} reviews"
        )

    st.divider()

    st.markdown("### What can you explore?")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="card">

            ### 📊 Business Analytics

            Ask questions about:

            - Revenue / GMV
            - Orders
            - Restaurants
            - Cuisines
            - Delivery performance
            - Cancellation rates

            **Powered by Text-to-SQL**

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            ### 💬 Customer Intelligence

            Ask questions about:

            - Customer complaints
            - Delivery feedback
            - Food quality
            - Pricing
            - Packaging
            - Service

            **Powered by RAG**

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PAGE 2 — TEXT TO SQL
# ============================================================

elif page == "📊 Ask Your Data":

    st.markdown(
        '<div class="main-title">'
        '📊 Ask Your Zomato Data'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Ask business questions in plain English.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Groq generates SQL → "
        "Snowflake executes it → "
        "results are displayed here."
    )

    question = st.text_input(
        "What would you like to know?",
        placeholder=(
            "e.g. Top 10 restaurants by revenue in Bangalore"
        )
    )

    with st.expander("💡 Example questions"):

        for q in EXAMPLE_QUESTIONS:

            st.markdown(f"- {q}")

    if question:

        with st.spinner("Generating SQL..."):

            try:

                sql = generate_sql(
                    question
                )

                st.subheader(
                    "Generated SQL"
                )

                st.code(
                    sql,
                    language="sql"
                )

                if not is_safe(sql):

                    st.error(
                        "The generated SQL "
                        "was blocked for safety."
                    )

                else:

                    with st.spinner(
                        "Querying Snowflake..."
                    ):

                        df = run_query(
                            sql
                        )

                    st.success(
                        f"{len(df)} rows returned"
                    )

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    if (
                        len(df.columns) == 2
                        and pd.api.types.is_numeric_dtype(
                            df.iloc[:, 1]
                        )
                    ):

                        st.subheader(
                            "📈 Visualization"
                        )

                        st.bar_chart(
                            df,
                            x=df.columns[0],
                            y=df.columns[1]
                        )

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# ============================================================
# PAGE 3 — RAG
# ============================================================

elif page == "💬 Customer Reviews":

    st.markdown(
        '<div class="main-title">'
        '💬 Customer Review Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Ask questions about what customers are saying.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"Searching {NEW_REVIEWS} reviews using "
        f"{EMBEDDING_MODEL} embeddings."
    )

    question = st.text_input(
        "What do customers say?",
        placeholder=(
            "e.g. What are the most common "
            "complaints about delivery?"
        )
    )

    if question:

        with st.spinner(
            "Searching customer reviews..."
        ):

            try:

                review_df = load_reviews()

                top_reviews = find_similar_reviews(
                    question,
                    review_df
                )

                answer = ask_llm(
                    question,
                    top_reviews
                )

                st.subheader(
                    "🤖 AI Answer"
                )

                st.write(answer)

                st.divider()

                st.subheader(
                    "🔎 Evidence"
                )

                st.caption(
                    "These are the reviews retrieved "
                    "as evidence for the answer."
                )

                evidence_df = top_reviews[
                    [
                        "city",
                        "rating",
                        "comment",
                        "score"
                    ]
                ].copy()

                evidence_df["score"] = (
                    evidence_df["score"]
                    .round(3)
                )

                st.dataframe(
                    evidence_df,
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# ============================================================
# PAGE 4 — ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="main-title">'
        'ℹ️ About the Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ## Zomato Analytics & GenAI Platform

        This project combines a modern analytics pipeline
        with Generative AI capabilities.

        ### Data Pipeline

        **AWS S3 → Snowflake → dbt → Airflow**

        ### Analytics

        - Fact and dimension modeling
        - Business marts
        - Delivery analytics
        - Restaurant performance
        - Revenue analysis

        ### Generative AI

        **Text-to-SQL**

        Natural language questions are converted into
        Snowflake SQL queries.

        **RAG**

        Customer review questions are answered using
        semantically relevant reviews.

        ### Technology

        - Python
        - SQL
        - Snowflake
        - dbt
        - Apache Airflow
        - AWS S3
        - Groq
        - Sentence Transformers
        - Streamlit
        - Power BI
        """
    )