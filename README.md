# 🍴 AI-Powered Zomato Analytics Platform

An end-to-end analytics and Generative AI project built using Zomato food-delivery data.

The project combines a data pipeline, analytical data modeling, customer-review intelligence, Retrieval-Augmented Generation (RAG), Natural Language-to-SQL, and an interactive Streamlit application.

🔗 **Live Demo:** https://zomato-data-pipeline.streamlit.app/

🔗 **GitHub:** https://github.com/Adarsh-ctrll/Zomato-Data-Pipeline

---

# 📌 Project Overview

Food-delivery platforms generate large amounts of transactional and customer-review data.

This project explores how that data can be transformed into useful business insights and then exposed through AI-powered interfaces.

The system has three major parts:

1. **Data Pipeline & Analytics**
2. **AI-powered Review Analysis**
3. **Natural Language Data Exploration**

The final application allows users to interact with the Zomato data using normal English instead of manually writing SQL.

---

# 🏗️ System Architecture

```text
                         ZOMATO DATA
                              │
                              ▼
                        Amazon S3
                              │
                              ▼
                     Apache Airflow
                              │
                              ▼
                         Snowflake
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
               RAW                       STAGING
                                            │
                                            ▼
                                           dbt
                                            │
                                            ▼
                                          MARTS
                                            │
                         ┌──────────────────┼──────────────────┐
                         │                  │                  │
                         ▼                  ▼                  ▼
                    Text-to-SQL            RAG          Review Enrichment
                         │                  │                  │
                         ▼                  ▼                  ▼
                    Snowflake        Review Retrieval      Groq LLM
                         │                  │                  │
                         └──────────────────┼──────────────────┘
                                            │
                                            ▼
                                      Streamlit App
