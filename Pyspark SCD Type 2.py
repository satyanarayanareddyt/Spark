#!/usr/bin/env python
# coding: utf-8

# ## Pyspark SCD Type 2
# 
# null

# In[1]:


from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ##### Load Source Data

# In[2]:


source_data = [
    ("C001", "John Miller", "Lincoln"),  
    ("C002", "Meera Patel", "Mumbai"),   
    ("C003", "Kabir Khan", "Chennai"),   
    ("C004", "Satyanarayanareddy Tadi", "Redmond")
]

source_columns = ["cust_id", "name", "address"]

source_df = spark.createDataFrame(source_data, source_columns)


# ##### Load Target Table

# In[3]:


target_table = "temp.dim_customer"

dim_delta = DeltaTable.forName(spark, target_table)
dim_df = dim_delta.toDF()
display(dim_df)


# ##### Read Current Dimension Records

# In[4]:


current_dim_df = (
    dim_df.filter(F.col("current_flag") == "Yes").
    select(
        "cust_key",
        "cust_id",
        "name",
        "address"
    )
)


# ##### Find New and Changed Customers

# In[10]:


changes_df = (
    source_df.alias("source")
    .join(
        current_dim_df.alias("target"),
        F.col("source.cust_id") == F.col("target.cust_id"),
        "left"
    )
    .filter(
        (F.col("target.cust_id").isNull()) |
        (
            (F.col("source.name") != F.col("target.name")) |
            (F.col("source.address") != F.col("target.address"))
        )
    )
    .select(
        F.col("source.cust_id").alias("cust_id"),
        F.col("source.name").alias("name"),
        F.col("source.address").alias("address"),
        F.col("target.cust_key").alias("existing_cust_key")
    )
)

display(changes_df)


# In[11]:


# ---------------------------------------------------------
# 4. Generate new surrogate keys
# ---------------------------------------------------------
# This assumes only one process writes to the dimension
# table at a time.

max_cust_key = (
    dim_df
    .agg(
        F.coalesce(
            F.max("cust_key"),
            F.lit(0)
        ).alias("max_cust_key")
    )
    .first()["max_cust_key"]
)
display(max_cust_key)

windowSpec = Window.orderBy(F.col("cust_id"))

changes_with_keys_df = (
    changes_df
    .withColumn(
        "new_cust_key",
        F.row_number().over(windowSpec) + max_cust_key
    )
)
display(changes_with_keys_df)



# In[12]:


# ---------------------------------------------------------
# 5. Rows for inserting new/current versions
# ---------------------------------------------------------

new_versions_df = (
    changes_with_keys_df
    .select(
        F.lit(None).cast("string").alias("merge_key"),
        F.col("new_cust_key").alias("cust_key"),
        "cust_id",
        "name",
        "address"
    )
)

display(new_versions_df)


# In[13]:


# ---------------------------------------------------------
# 6. Rows for expiring changed versions
# ---------------------------------------------------------

rows_to_expire_df = (
    changes_with_keys_df
    .filter(F.col("existing_cust_key").isNotNull())
    .select(
        F.col("cust_id").alias("merge_key"),
        F.col("new_cust_key").alias("cust_key"),
        "cust_id",
        "name",
        "address"
    )
)
display(rows_to_expire_df)


# In[15]:


# ---------------------------------------------------------
# 7. Stage rows for MERGE
# ---------------------------------------------------------

staged_df = new_versions_df.unionByName(rows_to_expire_df)
display(staged_df)


# ##### SCD Type 2 Merge

# In[16]:


future_end_date = "9999-12-31"


# In[19]:


(
    dim_delta.alias("target")
    .merge(
        staged_df.alias("source"),
        (F.col("target.cust_id") == F.col("source.merge_key")) & (F.col("target.current_flag") == "Yes")
    )
    .whenMatchedUpdate(
        set = {
            "end_date": F.current_date(),
            "current_flag": F.lit("No")
        }
    )
    .whenNotMatchedInsert(
        condition= F.col("source.merge_key").isNull(),
        values={
            "cust_key": F.col("source.cust_key"),
            "cust_id": F.col("source.cust_id"),
            "name": F.col("source.name"),
            "address": F.col("source.address"),
            "start_date": F.current_date(),
            "end_date": F.to_date(F.lit(future_end_date)),
            "current_flag": F.lit("Yes")
        }
    )
    .execute()
)


# In[20]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %%sql
# SELECT * FROM temp.dim_customer

