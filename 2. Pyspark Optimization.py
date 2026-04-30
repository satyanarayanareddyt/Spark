# Databricks notebook source
# MAGIC %md
# MAGIC # READ (Scan) Optimizations
# MAGIC Reduce the amount of data loaded from storage into Spark.
# MAGIC | # | Optimization               | What It Skips                                      | Automatic?                               |
# MAGIC |---|---------------------------|----------------------------------------------------|-------------------------------------------|
# MAGIC | 1 | Partition Pruning         | Entire folders/partitions                          | Yes (if filter on partition column)       |
# MAGIC | 2 | Dynamic Partition Pruning | Folders, using filter values from a join          | Yes (Spark 3.0+)                          |
# MAGIC | 3 | Predicate Pushdown        | Rows at the scan layer                             | Yes (unless a UDF blocks pushdown)        |
# MAGIC | 4 | Column Pruning            | Unneeded columns in Parquet/ORC                    | Yes (if you avoid `SELECT *`)             |
# MAGIC | 5 | Data Skipping (min/max)   | Files whose value ranges don't match               | Yes (Delta)                               |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **1. Partition Pruning**
# MAGIC
# MAGIC **Problem**
# MAGIC By default, Spark reads ALL files in a table — even if the query only needs a small subset.  
# MAGIC A table with 365 daily partitions forces Spark to list and open files in all 365 folders even when you want just one day.
# MAGIC
# MAGIC **Concept**
# MAGIC When a table is partitioned (data stored in column=value/ folder structure),
# MAGIC Spark can use filters on the partition column to SKIP ENTIRE FOLDERS.
# MAGIC Spark resolves the filter at plan time, prunes non-matching partition
# MAGIC directories, and only opens files in matching folders.
# MAGIC
# MAGIC Internally:
# MAGIC   - /data/sales/country=US/part-00000.parquet
# MAGIC   - /data/sales/country=UK/part-00000.parquet
# MAGIC   - /data/sales/country=IN/part-00000.parquet
# MAGIC
# MAGIC Query: WHERE country = 'US'
# MAGIC - Spark reads ONLY /data/sales/country=US/  → skips UK and IN folders entirely.
# MAGIC
# MAGIC **Solution**
# MAGIC Filter directly on the partition column so Spark can prune partitions during the scan.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Partition on LOW-CARDINALITY, FREQUENTLY-FILTERED columns (date, region,
# MAGIC     status, country).  Ideal: 100s to low 1000s of distinct values.
# MAGIC   - Filter DIRECTLY on the partition column: WHERE date = '2025-01-01'
# MAGIC   - Verify pruning with .explain(True) — look for "PartitionFilters".
# MAGIC   - Combine with predicate pushdown on non-partition columns for max effect.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Partition on HIGH-CARDINALITY columns (user_id, transaction_id) —
# MAGIC     creates millions of tiny folders/files (small file problem).
# MAGIC   - Wrap the partition column in functions:
# MAGIC       WRONG:  WHERE year(date) = 2025        → full scan, no pruning
# MAGIC       RIGHT:  WHERE date >= '2025-01-01' AND date < '2026-01-01'
# MAGIC   - Use UDFs on partition columns in the filter — Spark can't push UDFs down.
# MAGIC   - Over-partition: more than ~10,000 partitions causes metadata overhead.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Tables > 1 GB that are repeatedly filtered on the same column(s).
# MAGIC   - ETL pipelines where new data arrives in time-based batches.
# MAGIC   - NOT useful if queries never filter on the partition column.
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_partition_pruning():
# MAGIC     """Demonstrates partition pruning with and without proper filtering."""
# MAGIC
# MAGIC     # --- Setup: Create a partitioned table ---
# MAGIC     data = [
# MAGIC         ("US", "2025-01-01", 100.0),
# MAGIC         ("US", "2025-01-02", 150.0),
# MAGIC         ("UK", "2025-01-01", 200.0),
# MAGIC         ("UK", "2025-01-02", 250.0),
# MAGIC         ("IN", "2025-01-01", 80.0),
# MAGIC         ("IN", "2025-01-02", 90.0),
# MAGIC     ]
# MAGIC     schema = StructType([
# MAGIC         StructField("country", StringType()),
# MAGIC         StructField("date", StringType()),
# MAGIC         StructField("revenue", DoubleType()),
# MAGIC     ])
# MAGIC     df = spark.createDataFrame(data, schema)
# MAGIC     df.write.mode("overwrite").partitionBy("country").parquet("/tmp/training/sales_partitioned")
# MAGIC
# MAGIC     # --- GOOD: Filter on partition column → Spark prunes partitions ---
# MAGIC     df_pruned = spark.read.parquet("/tmp/training/sales_partitioned") \
# MAGIC         .filter("country = 'US'")
# MAGIC
# MAGIC     print("=== GOOD: Partition Pruning Active ===")
# MAGIC     df_pruned.explain(True)
# MAGIC     # Look for: PartitionFilters: [isnotnull(country), (country = US)]
# MAGIC     # Spark reads ONLY the country=US folder.
# MAGIC
# MAGIC     # --- BAD: UDF on partition column → Spark CANNOT prune ---
# MAGIC     from pyspark.sql.functions import udf
# MAGIC
# MAGIC     @udf(StringType())
# MAGIC     def upper_country(c):
# MAGIC         return c.upper() if c else None
# MAGIC
# MAGIC     df_no_prune = spark.read.parquet("/tmp/training/sales_partitioned") \
# MAGIC         .filter(upper_country(F.col("country")) == "US")
# MAGIC
# MAGIC     print("\n=== BAD: UDF blocks partition pruning — full scan ===")
# MAGIC     df_no_prune.explain(True)
# MAGIC     # PartitionFilters will be empty — Spark scans ALL partitions.
# MAGIC
# MAGIC     # --- BAD: Function wrapping partition column ---
# MAGIC     df_bad = spark.read.parquet("/tmp/training/sales_partitioned") \
# MAGIC         .filter(F.substring(F.col("country"), 1, 2) == "US")
# MAGIC
# MAGIC     print("\n=== BAD: Function on partition column — full scan ===")
# MAGIC     df_bad.explain(True)
# MAGIC ```
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **2. Dynamic Partition Pruning (DPP)**
# MAGIC **Problem**
# MAGIC Standard partition pruning requires a LITERAL filter value in the query.
# MAGIC But often the filter comes from JOINING with another table:
# MAGIC
# MAGIC ```
# MAGIC     SELECT f.* FROM fact_sales f
# MAGIC     JOIN dim_date d ON f.date_key = d.date_key
# MAGIC     WHERE d.quarter = 'Q1-2025'
# MAGIC ```
# MAGIC
# MAGIC Without DPP, Spark performs a FULL SCAN of fact_sales because the filter
# MAGIC value isn't known until the dim_date side is evaluated.
# MAGIC
# MAGIC **Concept**
# MAGIC Dynamic Partition Pruning (DPP) resolves the filter from the dimension
# MAGIC table FIRST, then pushes those values as a runtime filter into the fact
# MAGIC table scan.
# MAGIC
# MAGIC Step-by-step:
# MAGIC   1. Spark evaluates: dim_date WHERE quarter = 'Q1-2025' → gets date_key list
# MAGIC   2. Spark injects those date_key values as a filter on the fact_sales scan
# MAGIC   3. fact_sales reads ONLY the matching date_key partitions
# MAGIC
# MAGIC This happens automatically in Spark 3.0+ when conditions are met.
# MAGIC
# MAGIC **Solution**
# MAGIC Ensure the fact table is partitioned on the join key, and the dimension table is small enough to be broadcast.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Partition the FACT table on the join key (the column used in the ON clause).
# MAGIC   - Keep dimension tables small (broadcastable) — DPP works best when Spark
# MAGIC     can broadcast the dimension side.
# MAGIC   - Verify with .explain(True) — look for "DynamicPruningExpression".
# MAGIC   - Works automatically — no code change needed beyond proper partitioning.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Expect DPP to work if the fact table is NOT partitioned on the join key.
# MAGIC   - Disable it: spark.sql.optimizer.dynamicPartitionPruning.enabled = false.
# MAGIC   - Use DPP as a substitute for direct partition pruning — if you know the
# MAGIC     literal value, use it directly (faster plan, no dependency on dim query).
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Star/snowflake schema queries: fact table joined with dimension tables.
# MAGIC   - Any join where one side is small and the other is large + partitioned.
# MAGIC   - NOT useful when both tables are large and neither is partitioned.
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC def demo_dynamic_partition_pruning():
# MAGIC     """Demonstrates DPP with a fact-dimension join."""
# MAGIC
# MAGIC     # --- Setup: Partitioned fact table ---
# MAGIC     fact_data = [
# MAGIC         ("2025-01-01", "prod_1", 100.0),
# MAGIC         ("2025-01-02", "prod_2", 200.0),
# MAGIC         ("2025-04-15", "prod_1", 300.0),
# MAGIC         ("2025-04-16", "prod_3", 400.0),
# MAGIC         ("2025-07-01", "prod_2", 500.0),
# MAGIC     ]
# MAGIC     fact_schema = StructType([
# MAGIC         StructField("date_key", StringType()),
# MAGIC         StructField("product_id", StringType()),
# MAGIC         StructField("revenue", DoubleType()),
# MAGIC     ])
# MAGIC     fact_df = spark.createDataFrame(fact_data, fact_schema)
# MAGIC     fact_df.write.mode("overwrite") \
# MAGIC         .partitionBy("date_key") \
# MAGIC         .parquet("/tmp/training/fact_sales")
# MAGIC
# MAGIC     # --- Setup: Small dimension table ---
# MAGIC     dim_data = [
# MAGIC         ("2025-01-01", "Q1-2025"),
# MAGIC         ("2025-01-02", "Q1-2025"),
# MAGIC         ("2025-04-15", "Q2-2025"),
# MAGIC         ("2025-04-16", "Q2-2025"),
# MAGIC         ("2025-07-01", "Q3-2025"),
# MAGIC     ]
# MAGIC     dim_schema = StructType([
# MAGIC         StructField("date_key", StringType()),
# MAGIC         StructField("quarter", StringType()),
# MAGIC     ])
# MAGIC     dim_df = spark.createDataFrame(dim_data, dim_schema)
# MAGIC
# MAGIC     # --- DPP in action ---
# MAGIC     # Spark resolves Q1-2025 dates from dim_df, then prunes fact_sales partitions
# MAGIC     result = spark.read.parquet("/tmp/training/fact_sales") \
# MAGIC         .join(F.broadcast(dim_df), "date_key") \
# MAGIC         .filter("quarter = 'Q1-2025'")
# MAGIC
# MAGIC     print("=== Dynamic Partition Pruning ===")
# MAGIC     result.explain(True)
# MAGIC     # Look for: DynamicPruningExpression in the fact_sales scan
# MAGIC     result.show()
# MAGIC     
# MAGIC     ### Note: By default, DPP is enabled from Sprak 3.x and below spark 3 DPP won't exists. If someone disabled, use below code to enable it.
# MAGIC     spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
# MAGIC ```
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **3. Predicate Pushdown**
# MAGIC
# MAGIC **Problem**
# MAGIC - Without pushdown, Spark loads ALL rows from a file into memory, THEN applies the WHERE filter.  On a 10 GB Parquet file where only 1% of rows match, you waste 9.9 GB of I/O and memory.
# MAGIC
# MAGIC **Concept**
# MAGIC - Predicate pushdown moves the WHERE filter DOWN INTO the data source reader.
# MAGIC - For Parquet/ORC, the filter is evaluated at the row-group level using min/max statistics — entire row groups are skipped without decoding rows.
# MAGIC - For JDBC sources, the filter becomes part of the SQL query sent to the database.
# MAGIC
# MAGIC Pipeline WITHOUT pushdown: Read all rows → Load into memory → Apply filter → Output
# MAGIC
# MAGIC Pipeline WITH pushdown: Read file metadata → Skip non-matching row groups → Decode only matching rows → Output
# MAGIC
# MAGIC **Solution**
# MAGIC Use built-in Spark functions in filters.  Avoid Python UDFs, complex expressions, and non-deterministic functions in WHERE clauses.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Use built-in functions: col("x") == "val", col("x").between(1, 10),
# MAGIC     col("x").isin(["a", "b"]), col("x").isNull().
# MAGIC   - Filter as EARLY as possible in the DataFrame chain.
# MAGIC   - Verify with .explain(True) — look for "PushedFilters:" in the Scan node.
# MAGIC   - For JDBC sources, pass filters via .option("pushDownPredicate", "true")
# MAGIC     and verify the query sent to the database.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Use Python UDFs in WHERE clauses — they are opaque to the optimizer;
# MAGIC     Spark cannot push them to the Parquet reader.
# MAGIC   - Use non-deterministic functions (rand(), **current_timestamp()**) in filters —
# MAGIC     Spark won't push these down since results can vary.
# MAGIC   - Assume complex expressions always push down — compound expressions like
# MAGIC     (col("a") + col("b") > 10) may not push depending on the source.
# MAGIC   - Filter AFTER a shuffle/join when you could have filtered BEFORE.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Always.  This is the single most impactful automatic optimization.
# MAGIC   - Especially important for large Parquet/ORC tables and JDBC reads.
# MAGIC   - Less relevant for CSV/JSON (no row-group stats to skip).
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC def demo_predicate_pushdown():
# MAGIC     """Demonstrates predicate pushdown with Parquet files."""
# MAGIC
# MAGIC     # --- Setup ---
# MAGIC     data = [(i, f"product_{i % 100}", float(i * 10), "active" if i % 3 == 0 else "inactive")
# MAGIC             for i in range(10000)]
# MAGIC     schema = StructType([
# MAGIC         StructField("id", IntegerType()),
# MAGIC         StructField("product", StringType()),
# MAGIC         StructField("amount", DoubleType()),
# MAGIC         StructField("status", StringType()),
# MAGIC     ])
# MAGIC     df = spark.createDataFrame(data, schema)
# MAGIC     df.write.mode("overwrite").parquet("/tmp/training/pushdown_demo")
# MAGIC
# MAGIC     # --- GOOD: Built-in filter → pushed down to Parquet reader ---
# MAGIC     df_good = spark.read.parquet("/tmp/training/pushdown_demo") \
# MAGIC         .filter((F.col("status") == "active") & (F.col("amount") > 5000))
# MAGIC
# MAGIC     print("=== GOOD: Predicate Pushdown Active ===")
# MAGIC     df_good.explain(True)
# MAGIC     # Look for: PushedFilters: [IsNotNull(status), EqualTo(status,active),
# MAGIC     #                           GreaterThan(amount,5000.0)]
# MAGIC
# MAGIC     # --- BAD: Python UDF blocks pushdown ---
# MAGIC     from pyspark.sql.functions import udf
# MAGIC
# MAGIC     @udf(StringType())
# MAGIC     def check_status(s):
# MAGIC         return "yes" if s == "active" else "no"
# MAGIC
# MAGIC     df_bad = spark.read.parquet("/tmp/training/pushdown_demo") \
# MAGIC         .filter(check_status(F.col("status")) == "yes")
# MAGIC
# MAGIC     print("\n=== BAD: UDF Blocks Pushdown — Full Scan ===")
# MAGIC     df_bad.explain(True)
# MAGIC     # PushedFilters will be EMPTY — every row is read then filtered in Python.
# MAGIC
# MAGIC     # --- PATTERN: Filter BEFORE join, not after ---
# MAGIC     other_df = spark.range(100).withColumn("product", F.concat(F.lit("product_"), F.col("id")))
# MAGIC
# MAGIC     # BAD: filter after join
# MAGIC     result_bad = spark.read.parquet("/tmp/training/pushdown_demo") \
# MAGIC         .join(other_df, "product") \
# MAGIC         .filter("status = 'active'")
# MAGIC
# MAGIC     # GOOD: filter before join — fewer rows enter the shuffle
# MAGIC     result_good = spark.read.parquet("/tmp/training/pushdown_demo") \
# MAGIC         .filter("status = 'active'") \
# MAGIC         .join(other_df, "product")
# MAGIC
# MAGIC     print("\n=== GOOD: Filter BEFORE join ===")
# MAGIC     result_good.explain(True)
# MAGIC ```
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **4. Column Pruning (Project Pushdown)**
# MAGIC **Problem**
# MAGIC - Using SELECT * on a wide table (100+ columns) forces Spark to read EVERY column from Parquet, even if the query only needs 2.  On columnar formats, this is a massive waste — Parquet stores each column in separate column chunks, so unneeded columns can be entirely skipped.
# MAGIC
# MAGIC **Concept**
# MAGIC - Parquet/ORC files are columnar: data is stored column-by-column, not row-by-row.  When you SELECT only specific columns, Spark's Parquet reader skips the byte ranges of all other columns — never reads them from disk.
# MAGIC
# MAGIC ```
# MAGIC     Table: 200 columns, 1 TB total
# MAGIC     SELECT col_a, col_b → reads ~10 GB (2/200 columns ≈ 1% of data)
# MAGIC     SELECT *            → reads 1 TB
# MAGIC ```
# MAGIC This is called PROJECTION PUSHDOWN — the column list (projection) is
# MAGIC pushed into the reader.
# MAGIC
# MAGIC **Solution**
# MAGIC - Always select only the columns you need.  Never use SELECT * in production.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Explicitly list columns: df.select("col_a", "col_b")
# MAGIC   - Select early in the chain — before joins, groupBys, or any transformation.
# MAGIC   - For nested structs, select only needed fields: df.select("address.city")
# MAGIC     (requires nested schema pruning — see section 5).
# MAGIC   - Use .drop() to remove a few unneeded columns from a mostly-needed schema.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Use SELECT * or df.select("*") in production code.
# MAGIC   - Read all columns then filter them down later in the chain — Spark's
# MAGIC     optimizer can sometimes push projections down, but don't rely on it
# MAGIC     for complex plans.
# MAGIC   - Assume CSV/JSON benefits equally — they are row-based; the entire row
# MAGIC     is read regardless (but Spark still avoids deserializing unused columns).
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Always.  There is no reason not to prune columns.
# MAGIC   - Impact scales with table width: 10 columns → minor; 200 columns → massive.
# MAGIC """
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_column_pruning():
# MAGIC     """Demonstrates column pruning impact on Parquet reads."""
# MAGIC
# MAGIC     # --- Setup: Wide table with many columns ---
# MAGIC     data = [(i,) + tuple(f"val_{j}_{i}" for j in range(20)) for i in range(1000)]
# MAGIC     columns = ["id"] + [f"col_{j}" for j in range(20)]
# MAGIC     df = spark.createDataFrame(data, columns)
# MAGIC     df.write.mode("overwrite").parquet("/tmp/training/wide_table")
# MAGIC
# MAGIC     # --- BAD: SELECT * reads all 21 columns ---
# MAGIC     df_bad = spark.read.parquet("/tmp/training/wide_table")
# MAGIC     result_bad = df_bad.groupBy("col_0").count()
# MAGIC
# MAGIC     print("=== BAD: SELECT * — reads all columns ===")
# MAGIC     result_bad.explain(True)
# MAGIC
# MAGIC     # --- GOOD: Select only needed columns — reads 2 columns ---
# MAGIC     df_good = spark.read.parquet("/tmp/training/wide_table") \
# MAGIC         .select("id", "col_0")
# MAGIC     result_good = df_good.groupBy("col_0").count()
# MAGIC
# MAGIC     print("\n=== GOOD: Column Pruning — reads only id and col_0 ===")
# MAGIC     result_good.explain(True)
# MAGIC     # In the Scan node, ReadSchema will show only the selected columns.
# MAGIC ```
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **5. Data Skipping (Min/Max Statistics)**
# MAGIC **Problem**
# MAGIC Even after partition pruning, Spark may still open EVERY file within a
# MAGIC partition.  A partition with 1,000 files must open all 1,000 even if only
# MAGIC 1 file contains matching rows.
# MAGIC
# MAGIC **Concept**
# MAGIC Parquet files store MIN/MAX statistics per column per ROW GROUP (typically
# MAGIC 128 MB chunks).  Delta Lake extends this by storing per-FILE min/max in
# MAGIC the transaction log.
# MAGIC
# MAGIC When a filter like WHERE id BETWEEN 100 AND 200 is applied:
# MAGIC   - Parquet reader checks each row group's min/max for column "id"
# MAGIC   - If a row group's range is [500, 1000], it doesn't overlap [100, 200]
# MAGIC     → entire row group is SKIPPED without reading any data
# MAGIC   - Delta Lake does the same at FILE level using the transaction log
# MAGIC
# MAGIC EFFECTIVENESS depends on data ordering:
# MAGIC   - SORTED data → tight min/max ranges → excellent skipping
# MAGIC   - RANDOM data → wide min/max ranges → no skipping
# MAGIC     Example: File 1 has id [1, 100000] → overlaps almost any filter
# MAGIC
# MAGIC **Solution**
# MAGIC For Parquet: Write data sorted by frequently-filtered columns.
# MAGIC For Delta: Use OPTIMIZE + ZORDER BY or Liquid Clustering.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Sort data on the most frequently filtered column before writing.
# MAGIC   - For Delta tables, run OPTIMIZE with ZORDER BY on filter columns.
# MAGIC   - Place frequently filtered columns in the FIRST 32 positions — Delta
# MAGIC     collects stats only on the first 32 columns by default.
# MAGIC   - Use range filters (BETWEEN, >, <) — they benefit most from min/max.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Expect data skipping on randomly ordered data — min/max will span the
# MAGIC     full range of values, so no files can be skipped.
# MAGIC   - Rely on skipping for string columns with long values — stats are
# MAGIC     truncated (default 32 chars) and may not be useful.
# MAGIC   - Forget to run OPTIMIZE periodically — new appended files won't be
# MAGIC     co-located until compacted.
# MAGIC   - Put filter columns beyond position 32 in the schema (for Delta).
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Large Delta tables with range filters on ordered/clustered columns.
# MAGIC   - Timestamp, date, ID columns with natural ordering.
# MAGIC   - Less useful for boolean or low-cardinality columns (min=false, max=true
# MAGIC     covers everything).
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_data_skipping():
# MAGIC     """Demonstrates data skipping with sorted vs. unsorted data."""
# MAGIC
# MAGIC     # --- Setup: Generate data ---
# MAGIC     from pyspark.sql.functions import rand
# MAGIC
# MAGIC     df = spark.range(0, 100000).withColumn("value", (F.col("id") * 3.14).cast("double"))
# MAGIC
# MAGIC     # --- Sorted write: tight min/max per file → good skipping ---
# MAGIC     df.orderBy("id").repartition(10).write.mode("overwrite") \
# MAGIC         .parquet("/tmp/training/sorted_data")
# MAGIC
# MAGIC     # --- Random write: wide min/max per file → poor skipping ---
# MAGIC     df.orderBy(rand()).repartition(10).write.mode("overwrite") \
# MAGIC         .parquet("/tmp/training/random_data")
# MAGIC
# MAGIC     # --- Query both with a range filter ---
# MAGIC     print("=== Sorted Data: Data Skipping Effective ===")
# MAGIC     df_sorted = spark.read.parquet("/tmp/training/sorted_data") \
# MAGIC         .filter("id BETWEEN 100 AND 200")
# MAGIC     df_sorted.explain(True)
# MAGIC     # PushedFilters will show the range predicate.
# MAGIC     # On sorted data, most files will be skipped via min/max.
# MAGIC
# MAGIC     print("\n=== Random Data: Data Skipping Ineffective ===")
# MAGIC     df_random = spark.read.parquet("/tmp/training/random_data") \
# MAGIC         .filter("id BETWEEN 100 AND 200")
# MAGIC     df_random.explain(True)
# MAGIC     # Same pushed filter, but ALL files will be opened because
# MAGIC     # each file's min/max range spans nearly [0, 100000].
# MAGIC
# MAGIC     # --- Delta Lake ZORDER example (requires Delta) ---
# MAGIC     # Uncomment if running on Databricks or with delta-spark:
# MAGIC     #
# MAGIC     # df.write.mode("overwrite").format("delta").save("/tmp/training/delta_table")
# MAGIC     # spark.sql("OPTIMIZE delta.`/tmp/training/delta_table` ZORDER BY (id)")
# MAGIC     # # Now id values are co-located → tight min/max → excellent skipping
# MAGIC ```
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # TRANSFORMATION (Compute) Optimizations
# MAGIC Make operations like joins, aggregations, and UDFs faster.
# MAGIC
# MAGIC | #  | Optimization                      | Category        | What It Fixes                                      |
# MAGIC |----|-----------------------------------|-----------------|----------------------------------------------------|
# MAGIC |1| AQE — Join Conversion             | Join            | Switches Sort-Merge → Broadcast at runtime         |
# MAGIC |2| AQE — Skew Handling               | Join / Skew     | Splits skewed partitions into smaller tasks        |
# MAGIC |3| AQE — Coalescing Partitions       | Shuffle         | Merges too-many small post-shuffle partitions      |
# MAGIC |4| Broadcast Join                    | Join            | Eliminates shuffle by broadcasting small table     |
# MAGIC |5| Salting                           | Join / Skew     | Breaks hot keys into sub-keys for even distribution|
# MAGIC |6| Shuffle Partitions Tuning         | Shuffle         | Sets right partition count for data size           |
# MAGIC |7| Repartition                       | Shuffle         | Redistributes data evenly or by key                |
# MAGIC |8| Coalesce                          | Shuffle         | Reduces partitions without full shuffle            |

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1. AQE — JOIN CONVERSION (Sort-Merge → Broadcast at Runtime)
# MAGIC **Problem**
# MAGIC At PLAN TIME, Spark estimates table sizes using catalog statistics or file
# MAGIC sizes.  If stats are missing, stale, or inaccurate, Spark may choose a
# MAGIC Sort-Merge Join (SMJ) for a join where one side is actually small enough
# MAGIC to broadcast — wasting time on an expensive shuffle of BOTH sides.
# MAGIC
# MAGIC Example:
# MAGIC -   Table A: catalog says 500 MB  →  Spark picks Sort-Merge Join
# MAGIC -   Reality: after filter, Table A is only 8 MB  →  should be Broadcast
# MAGIC
# MAGIC Without AQE: both tables get shuffled (network transfer + disk I/O).
# MAGIC With AQE: after the filter stage completes, Spark sees the REAL size is
# MAGIC 8 MB and SWITCHES to Broadcast Hash Join — no shuffle on the large side.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC Adaptive Query Execution (AQE) re-optimizes the plan BETWEEN STAGES.
# MAGIC After each stage completes, Spark has ACTUAL data sizes (not estimates).
# MAGIC
# MAGIC For join conversion:
# MAGIC   1. Stage 1 (filter + scan) completes → Spark sees actual output = 8 MB
# MAGIC   2. Spark checks: 8 MB < autoBroadcastJoinThreshold (default 10 MB)?
# MAGIC   3. YES → switches from Sort-Merge Join to Broadcast Hash Join
# MAGIC   4. Large table side skips the shuffle entirely
# MAGIC
# MAGIC This is the MOST IMPACTFUL AQE feature because it eliminates shuffles
# MAGIC that were planned based on wrong estimates.
# MAGIC
# MAGIC **Solution**
# MAGIC Enable AQE (default in Spark 3.2+).  Ensure autoBroadcastJoinThreshold
# MAGIC is set appropriately.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Keep AQE enabled: spark.sql.adaptive.enabled = true (default in 3.2+).
# MAGIC   - Set a reasonable autoBroadcastJoinThreshold:
# MAGIC       Default: 10 MB.  Increase to 50-100 MB if executors have enough memory.
# MAGIC   - Filter tables BEFORE joins — AQE can only convert AFTER seeing the
# MAGIC     filtered size.  If you filter after the join, AQE sees the full size.
# MAGIC   - Run ANALYZE TABLE to give Spark better initial estimates — AQE then
# MAGIC     confirms/overrides at runtime.
# MAGIC   - Verify with .explain() or Spark UI SQL tab — look for
# MAGIC     "BroadcastHashJoin" instead of "SortMergeJoin".
# MAGIC
# MAGIC DON'T:
# MAGIC   - Disable AQE unless you have a specific Spark 2.x compatibility need.
# MAGIC   - Set autoBroadcastJoinThreshold too high (> 1 GB) — broadcasting a large
# MAGIC     table can cause driver/executor OOM.
# MAGIC   - Assume AQE always converts — it only converts when the ACTUAL post-stage
# MAGIC     size is below the threshold.  If both sides are large, SMJ is correct.
# MAGIC   - Rely solely on AQE without ANALYZE TABLE — AQE corrects bad plans at
# MAGIC     runtime, but good initial stats = better plan from the start.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Always enabled.  Zero downside in Spark 3.2+.
# MAGIC   - Most impactful when tables are heavily filtered before joins (real size
# MAGIC     is much smaller than raw table size).
# MAGIC   - Less impactful when both sides of the join are genuinely large.
# MAGIC """
# MAGIC
# MAGIC **Implementation:**
# MAGIC
# MAGIC ```
# MAGIC # Enable Adaptive Query Execution (AQE)
# MAGIC spark.conf.set("spark.sql.adaptive.enabled", "true")
# MAGIC
# MAGIC # Set auto broadcast join threshold (in bytes)
# MAGIC # Example: 10 MB
# MAGIC spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)
# MAGIC
# MAGIC # Optional: increase timeout
# MAGIC spark.conf.set("spark.sql.broadcastTimeout", 300)
# MAGIC
# MAGIC # For Disable
# MAGIC spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
# MAGIC
# MAGIC # Check Broadcast in Execution Plan
# MAGIC df.join(df2, "id").explain(True) 
# MAGIC Look for BroadcastHashJoin
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **2. AQE — SKEW HANDLING**
# MAGIC **Problem**
# MAGIC
# MAGIC Data skew means one join key has disproportionately more rows than others.
# MAGIC
# MAGIC     key = "US"  →  50 million rows  (one task handles ALL of these)
# MAGIC     key = "UK"  →  500K rows
# MAGIC     key = "IN"  →  800K rows
# MAGIC
# MAGIC In a Sort-Merge Join, all rows with the same key go to the SAME task.
# MAGIC The "US" task runs for hours while all other tasks finish in seconds.
# MAGIC This is called a STRAGGLER TASK — one slow task blocks the entire stage.
# MAGIC
# MAGIC Worse: the "US" task may run out of memory (OOM) because 50M rows don't
# MAGIC fit in one executor's memory.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC AQE detects skewed partitions AFTER the shuffle stage completes (it can
# MAGIC see the actual partition sizes).  When a partition is significantly larger
# MAGIC than the median:
# MAGIC
# MAGIC   1. AQE identifies the skewed partition (e.g., key = "US", 50M rows)
# MAGIC   2. It SPLITS the skewed partition into multiple smaller sub-partitions
# MAGIC   3. Each sub-partition is joined with a COPY of the matching data from
# MAGIC      the other side
# MAGIC   4. Results are combined — same output, but work is distributed
# MAGIC
# MAGIC Detection formula:
# MAGIC   Partition is skewed if:
# MAGIC     partition_size > skewedPartitionFactor × median_partition_size
# MAGIC     AND partition_size > skewedPartitionThresholdInBytes
# MAGIC
# MAGIC **Soultion**
# MAGIC
# MAGIC Enable AQE with skew join handling (default in Spark 3.2+).
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Keep AQE skew handling enabled:
# MAGIC       spark.sql.adaptive.skewJoin.enabled = true (default).
# MAGIC   - Tune detection thresholds for your data:
# MAGIC       skewedPartitionFactor = 5 (default) — partition must be 5x median
# MAGIC       skewedPartitionThresholdInBytes = 256MB (default) — minimum size to split
# MAGIC   - Check Spark UI SQL tab for "skew join optimization" in the plan.
# MAGIC   - Use AQE as the FIRST defense against skew — it's automatic.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Rely on AQE alone for extreme skew (1000:1 ratio) — manual salting
# MAGIC     may still be needed.
# MAGIC   - Set skewedPartitionFactor too low (e.g., 1.5) — it will split
# MAGIC     partitions that aren't truly skewed, adding overhead.
# MAGIC   - Set skewedPartitionThresholdInBytes too low — small partitions will be
# MAGIC     split unnecessarily.
# MAGIC   - Confuse AQE skew handling with salting — AQE is automatic at runtime;
# MAGIC     salting is a manual code technique.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Always enabled as a safety net.
# MAGIC   - Detects and fixes moderate skew automatically.
# MAGIC   - For extreme/known skew, combine with manual salting (see section 5).
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC # Enable AQE (mandatory)
# MAGIC spark.conf.set("spark.sql.adaptive.enabled", "true")
# MAGIC
# MAGIC # Enable skew join handling
# MAGIC spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
# MAGIC
# MAGIC # If a partition is 5x larger than median → considered skewed
# MAGIC spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
# MAGIC
# MAGIC # Minimum size to consider skew (default ~256MB)
# MAGIC spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", 256 * 1024 * 1024)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **3. AQE — COALESCING POST-SHUFFLE PARTITIONS**
# MAGIC **Problem**
# MAGIC Spark's default shuffle creates 200 partitions:
# MAGIC ```
# MAGIC   spark.sql.shuffle.partitions = 200 (default)
# MAGIC ```
# MAGIC
# MAGIC - After a shuffle (join, groupBy, window), data is redistributed into 200 partitions. If the data is small (e.g., 50 MB total), you get: 200 partitions × 250 KB each = 200 tiny tasks
# MAGIC - Each task has scheduling overhead (~50-100ms), so 200 tasks that each do 100ms of work spend more time on scheduling than on actual processing.
# MAGIC - The reverse: if data is 500 GB, 200 partitions means 2.5 GB per partition — too large, may OOM.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC AQE coalescing automatically MERGES small post-shuffle partitions into larger ones AFTER the shuffle completes (when actual sizes are known).
# MAGIC
# MAGIC Process:
# MAGIC   1. Shuffle writes 200 partitions
# MAGIC   2. AQE checks actual sizes of each partition
# MAGIC   3. Adjacent small partitions are merged until they reach the advisory
# MAGIC      target size (default 64 MB)
# MAGIC   4. 200 tiny partitions → maybe 5 appropriately-sized partitions
# MAGIC
# MAGIC This solves the "too many small partitions" problem WITHOUT you having to guess the right spark.sql.shuffle.partitions value upfront.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Enable AQE coalescing (default in Spark 3.2+).
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Keep AQE coalescing enabled:
# MAGIC       spark.sql.adaptive.coalescePartitions.enabled = true (default).
# MAGIC   - Set the advisory partition size:
# MAGIC       spark.sql.adaptive.advisoryPartitionSizeInBytes = 64MB (default)
# MAGIC       Increase to 128-256 MB for large datasets.
# MAGIC   - Still set spark.sql.shuffle.partitions to a reasonable UPPER BOUND
# MAGIC     (e.g., 200 or 2000) — AQE can only REDUCE, never increase.
# MAGIC   - Verify in Spark UI: check "number of partitions" in the stage after shuffle.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Rely on AQE to increase partitions — it only coalesces (merges).
# MAGIC     If you need MORE partitions, increase spark.sql.shuffle.partitions.
# MAGIC   - Set advisoryPartitionSizeInBytes too large (> 512 MB) — tasks may
# MAGIC     run out of memory.
# MAGIC   - Set spark.sql.shuffle.partitions = 1 thinking AQE will fix it —
# MAGIC     AQE can only merge, not split.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Always enabled.  Handles the common case of too many small partitions.
# MAGIC   - Especially useful when the same Spark job processes varying data sizes
# MAGIC     (some runs have 10 MB, others have 100 GB).
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC # Enable AQE (mandatory)
# MAGIC spark.conf.set("spark.sql.adaptive.enabled", "true")
# MAGIC
# MAGIC # Enable coalescing of partitions
# MAGIC spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
# MAGIC
# MAGIC # Target size of each partition after coalescing (default ~64MB)
# MAGIC spark.conf.set(
# MAGIC     "spark.sql.adaptive.advisoryPartitionSizeInBytes",
# MAGIC     64 * 1024 * 1024
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4. Broadcast Join
# MAGIC **Problem**
# MAGIC
# MAGIC In join operations, Spark typically performs a **shuffle join** (e.g., Sort-Merge Join):
# MAGIC
# MAGIC - Both tables are shuffled across the cluster
# MAGIC - Data movement is expensive (network + disk I/O)
# MAGIC - Performance degrades for large datasets
# MAGIC
# MAGIC Even if one table is small, Spark may still choose a shuffle join depending on configs.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC A **Broadcast Join** avoids shuffle by:
# MAGIC
# MAGIC 1. Sending (broadcasting) the **small table** to all executors
# MAGIC 2. Each executor joins its partition of the large table locally
# MAGIC
# MAGIC 👉 No shuffle required  
# MAGIC 👉 Much faster for small + large table joins  
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Manually force broadcast using `broadcast()`
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC
# MAGIC - Use when one table is small (few MBs to ~100MB depending on cluster)
# MAGIC - Use for dimension tables in fact-dimension joins
# MAGIC - Use when Spark is not automatically choosing broadcast
# MAGIC - Verify using .explain(True) → look for BroadcastHashJoin
# MAGIC
# MAGIC DON'T:
# MAGIC
# MAGIC - Don’t broadcast large tables → can cause OOM
# MAGIC - Don’t blindly broadcast without checking size
# MAGIC - Don’t use if both tables are large
# MAGIC
# MAGIC When to Use
# MAGIC
# MAGIC - Fact (large) + Dimension (small) joins
# MAGIC - Lookup tables
# MAGIC - Enrichment joins
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC from pyspark.sql.functions import broadcast
# MAGIC
# MAGIC result = large_df.join(
# MAGIC     broadcast(small_df),
# MAGIC     "join_key"
# MAGIC ) 
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **5. SALTING (Manual Skew Fix)**
# MAGIC
# MAGIC **Problem**
# MAGIC One join key has disproportionately more rows than others (data skew). AQE skew handling helps, but for EXTREME skew (1000:1 ratio) or when you KNOW which keys are hot, manual salting gives more control.
# MAGIC
# MAGIC     - key = "US"   →  50 million rows (one executor overwhelmed → OOM)
# MAGIC     - key = "UK"   →  500K rows
# MAGIC     - key = "IN"   →  800K rows
# MAGIC
# MAGIC The executor handling "US" runs out of memory or takes hours while all other tasks finish in seconds.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC Salting is a **manual technique** to distribute skewed data *before shuffle*.
# MAGIC
# MAGIC 👉 Idea:
# MAGIC Break one large key into multiple smaller keys
# MAGIC
# MAGIC Step-by-step:
# MAGIC   1. Add a random salt (0 to N-1) to the LARGE table's join key:
# MAGIC      "US" → "US_0", "US_1", ..., "US_9"   (10 sub-partitions)
# MAGIC   2. EXPLODE the SMALL table to match ALL salt values:
# MAGIC      "US" → "US_0", "US_1", ..., "US_9"   (10 copies of lookup row)
# MAGIC   3. Join on the salted key → data is evenly distributed
# MAGIC   4. Drop the salt columns
# MAGIC
# MAGIC BEFORE:  1 task handles 50M rows for "US"
# MAGIC AFTER:   10 tasks each handle 5M rows for "US_0" through "US_9"
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Add salt to the large table, explode the small table, join on salted key.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Choose salt count based on skew ratio:
# MAGIC       10 salts for 10:1 skew, 100 for 100:1, etc.
# MAGIC   - Salt ONLY the skewed keys if you know them — don't salt everything.
# MAGIC   - Use AQE as the first line of defense; add salting only when AQE
# MAGIC     isn't enough (extreme skew, known hot keys).
# MAGIC   - Remove salt columns after the join to keep output clean.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Over-salt — 1000 salts on moderately skewed data creates overhead
# MAGIC     (exploding the small table 1000x).
# MAGIC   - Forget to explode the small table — if you only salt the large side,
# MAGIC     keys won't match.
# MAGIC   - Use salting when broadcast join works — if the small table fits in
# MAGIC     memory, broadcast eliminates the problem entirely.
# MAGIC   - Apply salting blindly — profile your data first to identify which
# MAGIC     keys are actually skewed.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Known hot keys with extreme skew that AQE can't fully resolve.
# MAGIC   - Both tables are too large to broadcast.
# MAGIC   - After confirming skew in Spark UI (one task 100x slower than median).
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC def demo_salting():
# MAGIC     """Demonstrates salting technique for skewed joins."""
# MAGIC
# MAGIC     from pyspark.sql import functions as F
# MAGIC     from pyspark.sql.functions import explode, array, lit
# MAGIC
# MAGIC     # --- Setup: Skewed large table (90% hot_key) ---
# MAGIC     skewed_data = (
# MAGIC         [(i, "hot_key", float(i)) for i in range(90000)] +
# MAGIC         [(i, f"key_{i % 100}", float(i)) for i in range(90000, 100000)]
# MAGIC     )
# MAGIC     large_df = spark.createDataFrame(skewed_data, ["id", "join_key", "amount"])
# MAGIC
# MAGIC     # Small lookup table
# MAGIC     lookup_data = [("hot_key", "Hot Category", 1.5)] + \
# MAGIC                   [(f"key_{i}", f"Category_{i}", float(i)/100) for i in range(100)]
# MAGIC     small_df = spark.createDataFrame(lookup_data, ["join_key", "category", "multiplier"])
# MAGIC
# MAGIC     # --- BAD: Direct join ---
# MAGIC     print("=== BAD: Direct join (skew issue) ===")
# MAGIC     result_bad = large_df.join(small_df, "join_key")
# MAGIC     result_bad.explain(True)
# MAGIC
# MAGIC     # --- GOOD: Salted join ---
# MAGIC     num_salts = 10
# MAGIC
# MAGIC     # Step 1: Salt large table
# MAGIC     large_salted = large_df.withColumn(
# MAGIC         "salt", F.floor(F.rand() * num_salts).cast("int")
# MAGIC     ).withColumn(
# MAGIC         "salted_key", F.concat_ws("_", F.col("join_key"), F.col("salt"))
# MAGIC     )
# MAGIC
# MAGIC     # Step 2: Expand small table
# MAGIC     salt_array = array([lit(i) for i in range(num_salts)])
# MAGIC
# MAGIC     small_exploded = small_df.withColumn(
# MAGIC         "salt", explode(salt_array)
# MAGIC     ).withColumn(
# MAGIC         "salted_key", F.concat_ws("_", F.col("join_key"), F.col("salt"))
# MAGIC     )
# MAGIC
# MAGIC     # Step 3: Join
# MAGIC     result_good = large_salted.join(small_exploded, "salted_key") \
# MAGIC         .drop("salt", "salted_key")
# MAGIC
# MAGIC     print("\n=== GOOD: Salted join ===")
# MAGIC     result_good.explain(True)
# MAGIC
# MAGIC     # Validate
# MAGIC     print(f"Direct join count: {result_bad.count()}")
# MAGIC     print(f"Salted join count: {result_good.count()}")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 6. Shuffle Partitions Tuning
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC Every shuffle (join, groupBy, window, distinct, repartition) redistributes data into spark.sql.shuffle.partitions number of partitions.  Default: 200.
# MAGIC
# MAGIC   TOO FEW partitions (200 for 500 GB data):
# MAGIC     500 GB / 200 = 2.5 GB per partition → OOM, disk spills
# MAGIC     Few tasks → under-utilizes cluster cores
# MAGIC
# MAGIC   TOO MANY partitions (200 for 50 MB data):
# MAGIC     50 MB / 200 = 250 KB per partition → scheduling overhead
# MAGIC     200 tiny tasks → more time scheduling than processing
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC This setting controls the number of partitions AFTER any shuffle operation. It does NOT affect input scan partitions (those are controlled by spark.sql.files.maxPartitionBytes).
# MAGIC
# MAGIC Ideal partition size: 100-200 MB after shuffle.
# MAGIC
# MAGIC     Data size after shuffle    Recommended partitions
# MAGIC     ─────────────────────────  ─────────────────────
# MAGIC     < 1 GB                     10-50
# MAGIC     1-10 GB                    50-200
# MAGIC     10-100 GB                  200-2000
# MAGIC     100 GB - 1 TB              2000-10000
# MAGIC     > 1 TB                     10000+
# MAGIC
# MAGIC With AQE coalescing enabled, set this to an UPPER BOUND — AQE will merge
# MAGIC small partitions down.  Without AQE, you must guess the right value.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Set based on your typical data size.  With AQE, err on the high side.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - With AQE: set to a generous upper bound (e.g., 2000 for multi-GB data).
# MAGIC     AQE will coalesce down to the right number.
# MAGIC   - Without AQE: calculate → total_shuffle_data / 128MB = partition_count.
# MAGIC   - Set per-job if data sizes vary:
# MAGIC       spark.conf.set("spark.sql.shuffle.partitions", "500")
# MAGIC   - Monitor in Spark UI: check partition sizes in the stage after shuffle.
# MAGIC     Target: 100-200 MB per partition.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Leave at 200 for large datasets (> 50 GB) — too few, causes OOM/spills.
# MAGIC   - Leave at 200 for small datasets (< 100 MB) — too many, wastes scheduling.
# MAGIC   - Set to 1 — all data goes to one partition (guaranteed OOM on large data).
# MAGIC   - Confuse with spark.sql.files.maxPartitionBytes (input scan) or
# MAGIC     df.repartition() (explicit repartition).
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Every Spark job.  This is the most commonly misconfigured setting.
# MAGIC   - Adjust based on your data size and cluster resources.
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC # -------------------------------
# MAGIC # Baseline: Set shuffle partitions
# MAGIC # -------------------------------
# MAGIC # Keep this reasonably high to allow parallelism
# MAGIC spark.conf.set("spark.sql.shuffle.partitions", 200)
# MAGIC
# MAGIC # -------------------------------
# MAGIC # Enable AQE (Adaptive Query Execution)
# MAGIC # -------------------------------
# MAGIC spark.conf.set("spark.sql.adaptive.enabled", "true")
# MAGIC
# MAGIC # -------------------------------
# MAGIC # Enable AQE Coalesce (reduce small partitions)
# MAGIC # -------------------------------
# MAGIC spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
# MAGIC
# MAGIC # Target partition size after coalescing (default ~64MB)
# MAGIC spark.conf.set(
# MAGIC     "spark.sql.adaptive.advisoryPartitionSizeInBytes",
# MAGIC     64 * 1024 * 1024
# MAGIC )
# MAGIC
# MAGIC # -------------------------------
# MAGIC # Enable AQE Skew Join Handling
# MAGIC # -------------------------------
# MAGIC spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
# MAGIC
# MAGIC # Skew detection configs (optional tuning)
# MAGIC spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
# MAGIC spark.conf.set(
# MAGIC     "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes",
# MAGIC     256 * 1024 * 1024
# MAGIC )
# MAGIC
# MAGIC # -------------------------------
# MAGIC # (Optional) Enable dynamic join conversion
# MAGIC # -------------------------------
# MAGIC # Allows Spark to convert sort-merge join → broadcast join at runtime
# MAGIC spark.conf.set("spark.sql.adaptive.join.enabled", "true")
# MAGIC
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **7. Repartition**
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC Data may be unevenly distributed across partitions after a read or transformation:
# MAGIC   - Partition 1: 500 MB (overloaded)
# MAGIC   - Partition 2: 10 MB
# MAGIC   - Partition 3: 5 MB
# MAGIC
# MAGIC Or you need to redistribute data BY KEY before a join to avoid a shuffle
# MAGIC at join time.
# MAGIC
# MAGIC Or you need MORE partitions to utilize all available cores (input scan
# MAGIC created 4 partitions but you have 100 cores).
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC repartition(n) creates EXACTLY n partitions by doing a FULL SHUFFLE — all data is redistributed evenly using round-robin or hash partitioning.
# MAGIC
# MAGIC Two modes:
# MAGIC   - repartition(n)           → round-robin: even distribution, random order
# MAGIC   - repartition(n, "col")    → hash: rows with same col value go to same partition
# MAGIC   - repartition("col")       → hash with default partition count
# MAGIC
# MAGIC FULL SHUFFLE means ALL data moves across the network.  This is expensive but sometimes necessary.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Use repartition to increase partitions, redistribute evenly, or organize by key.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - repartition(n) to INCREASE partition count (coalesce can only decrease).
# MAGIC   - repartition("join_key") BEFORE a join to pre-shuffle by the join key —
# MAGIC     can eliminate the shuffle at join time if both sides are repartitioned.
# MAGIC   - repartition(n) before write to control output file count.
# MAGIC   - Use when you need EVEN distribution (after a skew-inducing operation).
# MAGIC
# MAGIC DON'T:
# MAGIC   - Use repartition to DECREASE partitions — use coalesce instead
# MAGIC     (no shuffle needed to merge adjacent partitions).
# MAGIC   - Repartition unnecessarily — it's a full shuffle.  Only use when the
# MAGIC     redistribution provides a clear downstream benefit.
# MAGIC   - Repartition by a high-cardinality column with low partition count —
# MAGIC     e.g., repartition(5, "user_id") with 1M users → extreme hash collisions.
# MAGIC   - Repartition right before .count() or .show() — no benefit, just overhead.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Increasing parallelism: 4 input partitions → repartition(100) for 100 cores.
# MAGIC   - Pre-partitioning for join: both sides repartitioned by join key.
# MAGIC   - Controlling output files: repartition(10).write.parquet() → 10 files.
# MAGIC   - Fixing skew: uneven partitions → repartition(n) for even distribution.
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_repartition():
# MAGIC     """Demonstrates repartition use cases."""
# MAGIC
# MAGIC     df = spark.range(0, 100000) \
# MAGIC         .withColumn("key", (F.col("id") % 100).cast("string")) \
# MAGIC         .withColumn("value", (F.col("id") * 2.5).cast("double"))
# MAGIC
# MAGIC     # --- Check current partitions ---
# MAGIC     print(f"=== Initial partitions: {df.rdd.getNumPartitions()} ===")
# MAGIC
# MAGIC     # --- repartition(n): Even distribution (round-robin) ---
# MAGIC     df_even = df.repartition(20)
# MAGIC     print(f"After repartition(20): {df_even.rdd.getNumPartitions()} partitions")
# MAGIC
# MAGIC     # --- repartition("col"): Hash by column ---
# MAGIC     df_by_key = df.repartition("key")
# MAGIC     print(f"After repartition('key'): {df_by_key.rdd.getNumPartitions()} partitions")
# MAGIC     # Same key always goes to same partition — useful before joins.
# MAGIC
# MAGIC     # --- repartition(n, "col"): Hash by column with specific count ---
# MAGIC     df_by_key_10 = df.repartition(10, "key")
# MAGIC     print(f"After repartition(10, 'key'): {df_by_key_10.rdd.getNumPartitions()} partitions")
# MAGIC
# MAGIC     # --- Pre-join repartitioning to avoid shuffle at join time ---
# MAGIC     df1 = spark.range(100000).withColumn("join_key", (F.col("id") % 100).cast("int"))
# MAGIC     df2 = spark.range(1000).withColumnRenamed("id", "join_key")
# MAGIC
# MAGIC     # Pre-repartition both by join key
# MAGIC     df1_repartitioned = df1.repartition("join_key")
# MAGIC     df2_repartitioned = df2.repartition("join_key")
# MAGIC     result = df1_repartitioned.join(df2_repartitioned, "join_key")
# MAGIC
# MAGIC     print("\n=== Pre-repartitioned Join ===")
# MAGIC     result.explain(True)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 8. **Coalesce**
# MAGIC **Problem**
# MAGIC
# MAGIC After filtering or other operations, you may have too many partitions with very little data in each:
# MAGIC
# MAGIC     Before filter:  200 partitions × 100 MB each = 20 GB
# MAGIC     After filter:   200 partitions × 500 KB each = 100 MB total
# MAGIC
# MAGIC 200 partitions for 100 MB = 200 tiny tasks = scheduling overhead.
# MAGIC
# MAGIC Or: before writing, you want fewer output files — 200 partitions would
# MAGIC create 200 tiny files (small file problem).
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC coalesce(n) REDUCES partition count by merging adjacent partitions. It does NOT shuffle data — partitions are simply combined locally.
# MAGIC
# MAGIC     coalesce(10):  merge partitions [1-20] → partition 1
# MAGIC                    merge partitions [21-40] → partition 2
# MAGIC                    ...
# MAGIC
# MAGIC Because there's NO SHUFFLE, coalesce is much faster than repartition.
# MAGIC But the resulting partitions may be UNEVEN — if partition 1 had 500 MB and
# MAGIC partition 2 had 5 MB, merging them gives 505 MB (no rebalancing).
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Use coalesce to reduce partitions before writes or when data shrinks after filtering.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Use coalesce to REDUCE partitions (e.g., 200 → 10 before write).
# MAGIC   - Use before write to control output file count:
# MAGIC       df.coalesce(10).write.parquet("/output/")  → 10 files
# MAGIC   - Use after a filter that dramatically reduces data size.
# MAGIC   - Prefer coalesce over repartition when decreasing — no shuffle.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Use coalesce to INCREASE partitions — coalesce(100) on a 10-partition
# MAGIC     DataFrame gives 10 partitions (it cannot split, only merge).
# MAGIC   - Use coalesce when you need EVEN distribution — coalesce doesn't
# MAGIC     rebalance, so output partitions may be very uneven.
# MAGIC   - coalesce(1) on large data — creates one giant file/partition,
# MAGIC     no parallelism on read.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Reducing output file count before write.
# MAGIC   - After heavy filtering (200 → 10 partitions).
# MAGIC   - When you need fewer partitions WITHOUT the cost of a full shuffle.
# MAGIC   - NOT when you need even distribution (use repartition instead).
# MAGIC
# MAGIC **INTERVIEW TRAP:**
# MAGIC   "Can you use coalesce to increase partitions?"
# MAGIC   - ANSWER: No. coalesce can only reduce. Use repartition to increase.
# MAGIC
# MAGIC   "What's the difference between repartition and coalesce?"
# MAGIC   - ANSWER: repartition = full shuffle (can increase or decrease, even distribution)
# MAGIC           - coalesce = no shuffle (can only decrease, may be uneven)
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC def demo_coalesce():
# MAGIC     """Demonstrates coalesce for reducing partitions without shuffle."""
# MAGIC
# MAGIC     df = spark.range(0, 100000) \
# MAGIC         .withColumn("value", (F.col("id") * 2.5).cast("double"))
# MAGIC
# MAGIC     # --- Start with many partitions ---
# MAGIC     df_many = df.repartition(200)
# MAGIC     print(f"=== Starting partitions: {df_many.rdd.getNumPartitions()} ===")
# MAGIC
# MAGIC     # --- coalesce: reduce without shuffle ---
# MAGIC     df_coalesced = df_many.coalesce(10)
# MAGIC     print(f"After coalesce(10): {df_coalesced.rdd.getNumPartitions()} partitions")
# MAGIC
# MAGIC     # --- Verify: no Exchange (shuffle) node in plan ---
# MAGIC     print("\n=== coalesce(10) plan — NO Exchange node ===")
# MAGIC     df_coalesced.explain(True)
# MAGIC     # No "Exchange" node — coalesce is purely local merging.
# MAGIC
# MAGIC     # --- Compare: repartition(10) adds a shuffle ---
# MAGIC     df_repartitioned = df_many.repartition(10)
# MAGIC     print("\n=== repartition(10) plan — HAS Exchange node ===")
# MAGIC     df_repartitioned.explain(True)
# MAGIC     # "Exchange RoundRobinPartitioning(10)" — full shuffle.
# MAGIC
# MAGIC     # --- TRAP: coalesce cannot increase ---
# MAGIC     df_small = spark.range(100).repartition(5)
# MAGIC     df_try_increase = df_small.coalesce(20)
# MAGIC     print(f"\n=== TRAP: coalesce(20) on 5-partition DF → {df_try_increase.rdd.getNumPartitions()} partitions ===")
# MAGIC     # Still 5 — coalesce cannot increase.
# MAGIC
# MAGIC     # --- Common pattern: coalesce before write ---
# MAGIC     print("\n=== Pattern: coalesce before write ===")
# MAGIC     # df.filter("status = 'active'").coalesce(10).write.parquet("/output/")
# MAGIC     print("df.filter(...).coalesce(10).write.parquet('/output/')")
# MAGIC     print("Creates exactly 10 output files instead of 200 tiny files.")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # WRITE & STORAGE Optimizations
# MAGIC Control output file layout for future read performance.
# MAGIC
# MAGIC | # | Optimization               | When to Use                                     |
# MAGIC |---|---------------------------|-------------------------------------------------|
# MAGIC | 1 | Coalesce before write     | Reduce output files (no shuffle)                |
# MAGIC | 2 | Repartition before write  | Even file sizes or increase file count          |
# MAGIC | 3 | partitionBy("col")        | Enable partition pruning for downstream reads   |
# MAGIC | 4 | optimizeWrite (Delta)     | Auto-size files at write time                   |
# MAGIC | 5 | autoCompact (Delta)       | Auto-compact after frequent appends             |
# MAGIC | 6 | OPTIMIZE (Delta)          | Compact accumulated small files                 |
# MAGIC | 7 | ZORDER BY (Delta)         | Co-locate data for better data skipping         |
# MAGIC | 8 | V-Order (Fabric)          | Faster reads in Fabric engine                   |
# MAGIC | 9 | VACUUM (Delta)            | Remove old files and reclaim storage            |

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1. Coalesce Before Write
# MAGIC **Problem**
# MAGIC
# MAGIC After a shuffle or filter, your DataFrame may have 200 partitions with tiny data in each.  Each partition becomes ONE output file:
# MAGIC
# MAGIC     200 partitions → 200 files × 500 KB each = 100 MB in 200 tiny files
# MAGIC
# MAGIC Downstream readers suffer:
# MAGIC   - Slow file listing (200 LIST calls on cloud storage)
# MAGIC   - 200 tasks for 100 MB of data (scheduling overhead)
# MAGIC   - Poor compression (small files can't compress efficiently)
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC coalesce(n) merges adjacent partitions into fewer, larger partitions. WITHOUT a shuffle.  Fewer partitions = fewer output files.
# MAGIC
# MAGIC     df has 200 partitions → coalesce(10) → 10 partitions → write → 10 files
# MAGIC
# MAGIC No data moves across the network — partitions are merged locally on the same executor.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Add coalesce(n) just before .write to control output file count.
# MAGIC
# MAGIC Key Points
# MAGIC
# MAGIC DO:
# MAGIC   - Target output file size: 128 MB – 1 GB per file.
# MAGIC   - Calculate: total_data_size / target_file_size = number of files.
# MAGIC     Example: 5 GB data / 500 MB target = 10 files → coalesce(10)
# MAGIC   - Use AFTER filter/transformation that reduces data significantly.
# MAGIC
# MAGIC DON'T:
# MAGIC   - coalesce(1) on large data — one giant file, no read parallelism.
# MAGIC   - Use coalesce when data is UNEVENLY distributed — merged partitions
# MAGIC     will be uneven too.  Use repartition instead for even files.
# MAGIC   - Coalesce to MORE partitions than you have — it can only reduce.
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC def demo_coalesce_before_write():
# MAGIC     """Coalesce reduces output file count without shuffle."""
# MAGIC
# MAGIC     df = spark.range(0, 100000) \
# MAGIC         .withColumn("name", F.concat(F.lit("user_"), F.col("id").cast("string"))) \
# MAGIC         .withColumn("amount", (F.col("id") * 1.5).cast("double"))
# MAGIC
# MAGIC     print(f"Partitions before coalesce: {df.rdd.getNumPartitions()}")
# MAGIC
# MAGIC     # --- BAD: Many tiny output files ---
# MAGIC     df.write.mode("overwrite").parquet("/tmp/training/write_no_coalesce")
# MAGIC
# MAGIC     # --- GOOD: Controlled file count ---
# MAGIC     df.coalesce(5).write.mode("overwrite").parquet("/tmp/training/write_coalesced")
# MAGIC
# MAGIC     # Count output files
# MAGIC     import os
# MAGIC     bad_files = spark.read.parquet("/tmp/training/write_no_coalesce").inputFiles()
# MAGIC     good_files = spark.read.parquet("/tmp/training/write_coalesced").inputFiles()
# MAGIC     print(f"Without coalesce: {len(bad_files)} files")
# MAGIC     print(f"With coalesce(5): {len(good_files)} files")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **2. Repartition Before Write**
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC coalesce produces UNEVEN files because it merges without rebalancing. If partition 1 has 500 MB and partition 2 has 5 MB, coalescing them gives 505 MB — one huge file and one tiny file.
# MAGIC
# MAGIC Also, you may need to INCREASE file count (coalesce can only decrease).
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC repartition(n) does a FULL SHUFFLE to create exactly n EVENLY-SIZED
# MAGIC partitions.  Every output file will be roughly the same size.
# MAGIC
# MAGIC     repartition(10).write → 10 files, each ~same size
# MAGIC
# MAGIC Cost: full shuffle (all data moves across the network).
# MAGIC Benefit: perfectly even output files.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Use repartition before write when you need EVEN file sizes or need to INCREASE the partition count.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Use when output files MUST be evenly sized (downstream systems need it).
# MAGIC   - Use to INCREASE file count (coalesce can only decrease).
# MAGIC   - repartition(n, "col") to co-locate same-key rows in same files —
# MAGIC     improves downstream filter/join performance.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Use repartition when coalesce would work — repartition is slower
# MAGIC     (full shuffle vs. local merge).
# MAGIC   - Repartition to 1 — same problem as coalesce(1): one giant file.
# MAGIC   - Repartition by a high-cardinality column with low n — hash collisions
# MAGIC     make files uneven anyway.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   coalesce → reducing file count, unevenness is acceptable.
# MAGIC   repartition → need EXACT count with EVEN sizes, or need to increase.
# MAGIC
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC def demo_repartition_before_write():
# MAGIC     """Repartition creates evenly-sized output files."""
# MAGIC
# MAGIC     df = spark.range(0, 100000) \
# MAGIC         .withColumn("category", (F.col("id") % 5).cast("string")) \
# MAGIC         .withColumn("amount", (F.col("id") * 2.0).cast("double"))
# MAGIC
# MAGIC     # --- Repartition for even file sizes ---
# MAGIC     df.repartition(10).write.mode("overwrite") \
# MAGIC         .parquet("/tmp/training/write_repartitioned")
# MAGIC
# MAGIC     # --- Repartition by column — co-locates same category in same file ---
# MAGIC     df.repartition(5, "category").write.mode("overwrite") \
# MAGIC         .parquet("/tmp/training/write_repartitioned_by_col")
# MAGIC
# MAGIC     files = spark.read.parquet("/tmp/training/write_repartitioned").inputFiles()
# MAGIC     print(f"repartition(10): {len(files)} files (evenly sized)")
# MAGIC
# MAGIC     files_by_col = spark.read.parquet("/tmp/training/write_repartitioned_by_col").inputFiles()
# MAGIC     print(f"repartition(5, 'category'): {len(files_by_col)} files (grouped by category)")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3. Partition By (partitionBy)
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC A table has 1 TB of data across all dates.  Every query filters by date:
# MAGIC     WHERE date = '2025-01-15'
# MAGIC
# MAGIC Without partitioning, Spark reads ALL 1 TB — then throws away 99.7%.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC partitionBy("col") writes data into a FOLDER STRUCTURE based on column
# MAGIC values:
# MAGIC
# MAGIC     /output/date=2025-01-01/part-00000.parquet
# MAGIC     /output/date=2025-01-02/part-00000.parquet
# MAGIC     /output/date=2025-01-03/part-00000.parquet
# MAGIC
# MAGIC When a query filters WHERE date = '2025-01-15', Spark reads ONLY that
# MAGIC folder — skipping all other dates.  This is PARTITION PRUNING (see read
# MAGIC optimizations).
# MAGIC
# MAGIC partitionBy is a WRITE-TIME decision that enables READ-TIME optimization.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Partition by the column(s) most frequently used in WHERE filters.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Partition by LOW-CARDINALITY columns: date, region, status, country.
# MAGIC     Ideal: 100s to low 1000s of distinct values.
# MAGIC   - Partition by the column used in downstream WHERE filters.
# MAGIC   - Combine with coalesce/repartition to control files PER partition:
# MAGIC       df.repartition(1, "date").write.partitionBy("date").parquet(path)
# MAGIC       → 1 file per date partition.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Partition by HIGH-CARDINALITY columns (user_id, order_id) —
# MAGIC     millions of folders, each with one tiny file = SMALL FILE PROBLEM.
# MAGIC   - Partition by more than 2-3 columns — combinatorial explosion of folders.
# MAGIC     partitionBy("year", "month", "day", "hour") = way too many folders.
# MAGIC   - Partition by columns that aren't used in filters — write overhead
# MAGIC     with no read benefit.
# MAGIC   - Forget that partition columns are REMOVED from the Parquet file data
# MAGIC     (they're encoded in the folder path).
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Any table > 1 GB that is repeatedly filtered by the same column.
# MAGIC   - Time-series data: partition by date or year/month.
# MAGIC   - Multi-tenant data: partition by tenant_id (if low cardinality).
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC def demo_partition_by():
# MAGIC     """partitionBy creates folder structure for partition pruning."""
# MAGIC
# MAGIC     data = [(f"user_{i}", "2025-01-15" if i % 3 == 0 else "2025-01-16", float(i))
# MAGIC             for i in range(10000)]
# MAGIC     df = spark.createDataFrame(data, ["user", "date", "amount"])
# MAGIC
# MAGIC     # --- Write partitioned by date ---
# MAGIC     df.write.mode("overwrite") \
# MAGIC         .partitionBy("date") \
# MAGIC         .parquet("/tmp/training/write_partitioned")
# MAGIC
# MAGIC     # --- Read with filter → partition pruning ---
# MAGIC     result = spark.read.parquet("/tmp/training/write_partitioned") \
# MAGIC         .filter("date = '2025-01-15'")
# MAGIC
# MAGIC     print("=== partitionBy('date') → Partition Pruning on Read ===")
# MAGIC     result.explain(True)
# MAGIC     # Look for: PartitionFilters: [date = 2025-01-15]
# MAGIC     print(f"Rows for 2025-01-15: {result.count()}")
# MAGIC
# MAGIC     # --- Control files per partition ---
# MAGIC     df.repartition(1, "date").write.mode("overwrite") \
# MAGIC         .partitionBy("date") \
# MAGIC         .parquet("/tmp/training/write_partitioned_1file")
# MAGIC     print("\nrepartition(1, 'date') + partitionBy('date') → 1 file per date folder")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **4. Optimize Write (Delta Lake)**
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC - In Spark, each task writes ONE output file.  If you have 200 tasks but only 50 MB of data, you get 200 files × 250 KB each — tiny files.
# MAGIC - This happens even without explicit repartition/coalesce because the number of output files = number of tasks writing.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC optimizeWrite is a Delta Lake feature that automatically COALESCES partitions at write time.  It re-bins data so output files are close to the target size (~128 MB) regardless of how many tasks there are.
# MAGIC
# MAGIC     Without optimizeWrite:  200 tasks → 200 files (many tiny)
# MAGIC     With optimizeWrite:     200 tasks → Spark re-bins → ~5 files (right-sized)
# MAGIC
# MAGIC It acts like an automatic coalesce INSIDE the write operation.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Enable for Delta tables (automatic in Databricks, manual in OSS Spark).
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Enable for all Delta table writes:
# MAGIC       spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
# MAGIC     Or per-table: ALTER TABLE t SET TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true)
# MAGIC   - Use instead of manual coalesce for Delta tables — it handles sizing
# MAGIC     automatically.
# MAGIC   - Combine with autoCompact for full automation (see section 5).
# MAGIC
# MAGIC DON'T:
# MAGIC   - Use with explicit repartition/coalesce — they conflict.  Let
# MAGIC     optimizeWrite handle file sizing automatically.
# MAGIC   - Expect it to work on plain Parquet — Delta Lake feature only.
# MAGIC   - Assume it replaces OPTIMIZE — it only helps at WRITE time;
# MAGIC     existing small files still need OPTIMIZE to compact.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - All Delta table writes — no downside.
# MAGIC   - Especially for streaming or frequent small appends.
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC def demo_optimize_write():
# MAGIC     """optimizeWrite auto-sizes output files at write time (Delta)."""
# MAGIC
# MAGIC     # --- Delta Lake required — uncomment if available ---
# MAGIC     # spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
# MAGIC
# MAGIC     # df = spark.range(0, 100000) \
# MAGIC     #     .withColumn("value", (F.col("id") * 2.0).cast("double"))
# MAGIC     #
# MAGIC     # # Without optimizeWrite: many tasks → many small files
# MAGIC     # df.write.mode("overwrite").format("delta") \
# MAGIC     #     .save("/tmp/training/delta_no_optwrite")
# MAGIC     #
# MAGIC     # # With optimizeWrite: tasks auto-coalesced → fewer, right-sized files
# MAGIC     # df.write.mode("overwrite").format("delta") \
# MAGIC     #     .option("optimizeWrite", "true") \
# MAGIC     #     .save("/tmp/training/delta_optwrite")
# MAGIC
# MAGIC     print("=== optimizeWrite (Delta Lake) ===")
# MAGIC     print("Enable: spark.conf.set('spark.databricks.delta.optimizeWrite.enabled', 'true')")
# MAGIC     print("Or per-write: .option('optimizeWrite', 'true')")
# MAGIC     print("Effect: auto-coalesces partitions → fewer, right-sized output files")
# MAGIC     print("Use for: all Delta writes, especially streaming/frequent appends")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **5. Auto Compact (Delta Lake)**
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC Even with optimizeWrite, repeated APPEND operations accumulate files:
# MAGIC
# MAGIC     Day 1: append → 5 files
# MAGIC     Day 2: append → 5 more files (total: 10)
# MAGIC     Day 30: append → 5 more (total: 150 files)
# MAGIC
# MAGIC optimizeWrite controls per-write file count.  But across many writes,
# MAGIC files accumulate.  Eventually you have hundreds of small files.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC autoCompact triggers a MINI OPTIMIZE automatically after each write.
# MAGIC When the number of small files in a partition exceeds a threshold,
# MAGIC Spark compacts them into larger files.
# MAGIC
# MAGIC     After append: check → too many small files? → compact → done
# MAGIC
# MAGIC It runs in the SAME job, right after the write — no separate job needed.
# MAGIC
# MAGIC autoCompact does a LIGHTER compaction than full OPTIMIZE:
# MAGIC   - Only compacts files < 128 MB
# MAGIC   - Doesn't rewrite large files
# MAGIC   - Doesn't do ZORDER (just file sizing)
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Enable for Delta tables with frequent appends.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Enable with optimizeWrite for full automation:
# MAGIC       spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
# MAGIC     Or per-table: ALTER TABLE t SET TBLPROPERTIES (delta.autoOptimize.autoCompact = true)
# MAGIC   - Use for streaming sinks and incremental ETL that append frequently.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Rely on autoCompact alone for large tables — it does light compaction.
# MAGIC     Still run full OPTIMIZE periodically for best performance (see section 6).
# MAGIC   - Enable on tables that are rarely appended to — no benefit, just overhead.
# MAGIC   - Expect ZORDER from autoCompact — it only sizes files, doesn't reorganize
# MAGIC     data layout.  Use OPTIMIZE ZORDER BY for that.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Tables with frequent appends (streaming, hourly/daily ingestion).
# MAGIC   - Combine: optimizeWrite (per-write) + autoCompact (across writes).
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_auto_compact():
# MAGIC     """autoCompact triggers light compaction after each write (Delta)."""
# MAGIC
# MAGIC     # --- Delta Lake required ---
# MAGIC     # spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
# MAGIC
# MAGIC     # # Simulate frequent appends
# MAGIC     # for batch in range(5):
# MAGIC     #     df = spark.range(batch * 1000, (batch + 1) * 1000) \
# MAGIC     #         .withColumn("value", (F.col("id") * 1.5).cast("double"))
# MAGIC     #     df.write.mode("append").format("delta") \
# MAGIC     #         .save("/tmp/training/delta_autocompact")
# MAGIC     #     # autoCompact checks after each append and compacts if needed
# MAGIC
# MAGIC     print("=== autoCompact (Delta Lake) ===")
# MAGIC     print("Enable: spark.conf.set('spark.databricks.delta.autoCompact.enabled', 'true')")
# MAGIC     print("Effect: after each write, compacts small files if too many accumulate")
# MAGIC     print("Light compaction — only merges files < 128 MB, no ZORDER")
# MAGIC     print("Use for: streaming sinks, frequent appends")
# MAGIC     print("Still run full OPTIMIZE periodically for best read performance")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 6. Optimize (Delta Lake)
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC Over time, Delta tables accumulate many small files from:
# MAGIC   - Streaming micro-batches (every 30 seconds → hundreds of files/day)
# MAGIC   - Frequent appends (hourly ETL)
# MAGIC   - UPDATE/DELETE/MERGE operations (copy-on-write creates new files)
# MAGIC
# MAGIC Reading 10,000 small files is much slower than reading 100 right-sized
# MAGIC files because of:
# MAGIC   - File listing overhead (10,000 LIST calls on cloud storage)
# MAGIC   - Per-file open/close overhead
# MAGIC   - Poor compression (small files don't compress well)
# MAGIC   - Too many tasks in Spark
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC OPTIMIZE compacts small files into larger, optimally-sized files
# MAGIC (target: ~1 GB per file).
# MAGIC
# MAGIC     BEFORE: 10,000 files × 1 MB each = 10 GB
# MAGIC     AFTER:  10 files × 1 GB each = 10 GB  (same data, fewer files)
# MAGIC
# MAGIC Process:
# MAGIC   1. Read all small files in a partition
# MAGIC   2. Rewrite them into fewer, larger files
# MAGIC   3. Update the Delta transaction log
# MAGIC   4. Old files are marked for deletion (cleaned up by VACUUM)
# MAGIC
# MAGIC OPTIMIZE is IDEMPOTENT — running it twice has no effect if files are
# MAGIC already optimally sized.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Run OPTIMIZE periodically on Delta tables.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Run OPTIMIZE after bulk data loading completes.
# MAGIC   - Schedule OPTIMIZE as a maintenance job (daily or weekly).
# MAGIC   - Use WHERE clause to optimize specific partitions:
# MAGIC       OPTIMIZE table WHERE date = '2025-01-15'
# MAGIC   - Combine with ZORDER BY for data co-location (see section 7).
# MAGIC
# MAGIC DON'T:
# MAGIC   - Run OPTIMIZE during active writes — it competes for resources.
# MAGIC   - OPTIMIZE too frequently on append-heavy tables — each run rewrites
# MAGIC     files; use autoCompact for continuous light compaction.
# MAGIC   - Forget to VACUUM after OPTIMIZE — old files consume storage.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - After initial bulk load of a Delta table.
# MAGIC   - Periodically on tables with many small files.
# MAGIC   - Before running analytics on a table with accumulated appends.
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_optimize():
# MAGIC     """OPTIMIZE compacts small files into larger ones (Delta)."""
# MAGIC
# MAGIC     # --- Delta Lake required ---
# MAGIC     # # Create table with many small files (simulating micro-batches)
# MAGIC     # for i in range(20):
# MAGIC     #     df = spark.range(i * 500, (i + 1) * 500) \
# MAGIC     #         .withColumn("date", F.lit("2025-01-15")) \
# MAGIC     #         .withColumn("value", (F.col("id") * 1.5).cast("double"))
# MAGIC     #     df.write.mode("append").format("delta") \
# MAGIC     #         .partitionBy("date") \
# MAGIC     #         .save("/tmp/training/delta_optimize_demo")
# MAGIC     #
# MAGIC     # # Check file count BEFORE optimize
# MAGIC     # detail_before = spark.sql("DESCRIBE DETAIL delta.`/tmp/training/delta_optimize_demo`")
# MAGIC     # detail_before.select("numFiles").show()  # ~20 small files
# MAGIC     #
# MAGIC     # # --- OPTIMIZE: compact all files ---
# MAGIC     # spark.sql("OPTIMIZE delta.`/tmp/training/delta_optimize_demo`")
# MAGIC     #
# MAGIC     # # --- OPTIMIZE specific partition ---
# MAGIC     # spark.sql("""
# MAGIC     #     OPTIMIZE delta.`/tmp/training/delta_optimize_demo`
# MAGIC     #     WHERE date = '2025-01-15'
# MAGIC     # """)
# MAGIC     #
# MAGIC     # # Check file count AFTER optimize
# MAGIC     # detail_after = spark.sql("DESCRIBE DETAIL delta.`/tmp/training/delta_optimize_demo`")
# MAGIC     # detail_after.select("numFiles").show()  # ~1-2 right-sized files
# MAGIC
# MAGIC     print("=== OPTIMIZE (Delta Lake) ===")
# MAGIC     print("Compacts small files into larger ~1 GB files")
# MAGIC     print("")
# MAGIC     print("Usage:")
# MAGIC     print("  OPTIMIZE my_table                          -- full table")
# MAGIC     print("  OPTIMIZE my_table WHERE date = '2025-01-15' -- one partition")
# MAGIC     print("  OPTIMIZE my_table ZORDER BY (col)          -- compact + reorder (see #7)")
# MAGIC     print("")
# MAGIC     print("Schedule: daily/weekly depending on write frequency")
# MAGIC     print("Run AFTER bulk loads, BEFORE analytics")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **7. ZORDER BY**
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC OPTIMIZE compacts files but doesn't change how data is ORDERED inside
# MAGIC them.  If customer_id values are randomly distributed across files,
# MAGIC every file's min/max range for customer_id spans the full range:
# MAGIC
# MAGIC     File 1: customer_id min=1,    max=99999   ← wide range
# MAGIC     File 2: customer_id min=5,    max=99995   ← wide range
# MAGIC     File 3: customer_id min=10,   max=99998   ← wide range
# MAGIC
# MAGIC     Query: WHERE customer_id = 500
# MAGIC     Data skipping checks min/max → ALL files overlap → reads ALL files.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC ZORDER BY physically reorganizes data so that rows with SIMILAR values
# MAGIC of the specified column(s) are stored in the SAME files.
# MAGIC
# MAGIC     BEFORE ZORDER:
# MAGIC       File 1: customer_id [1 - 99999]      ← overlaps everything
# MAGIC       File 2: customer_id [5 - 99995]
# MAGIC       File 3: customer_id [10 - 99998]
# MAGIC
# MAGIC     AFTER ZORDER BY (customer_id):
# MAGIC       File 1: customer_id [1 - 33000]      ← tight, non-overlapping
# MAGIC       File 2: customer_id [33001 - 66000]
# MAGIC       File 3: customer_id [66001 - 99999]
# MAGIC
# MAGIC     Query: WHERE customer_id = 500
# MAGIC     Data skipping: only File 1 overlaps → reads 1 file instead of 3.
# MAGIC
# MAGIC ZORDER uses a space-filling curve (Z-order curve) that maps multi-
# MAGIC dimensional data into a linear order while preserving locality.  This
# MAGIC means you can ZORDER BY (col_a, col_b) and get good skipping on BOTH.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Run OPTIMIZE with ZORDER BY on columns used in WHERE filters.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - ZORDER BY the 1-2 columns most frequently used in WHERE filters.
# MAGIC   - Combine with OPTIMIZE — ZORDER is an option of OPTIMIZE, not separate.
# MAGIC   - Prefer ZORDER on high-cardinality columns where min/max skipping is
# MAGIC     poor without it (customer_id, user_id, timestamp).
# MAGIC   - Re-run periodically as new data is appended (new files won't be
# MAGIC     ZORDERed until the next OPTIMIZE).
# MAGIC
# MAGIC DON'T:
# MAGIC   - ZORDER BY more than 3-4 columns — effectiveness drops sharply.
# MAGIC     The Z-order curve can only preserve locality in a few dimensions.
# MAGIC   - ZORDER BY low-cardinality columns (status, country) — min/max
# MAGIC     skipping already works well on these.
# MAGIC   - ZORDER BY the partition column — it's redundant (partition pruning
# MAGIC     already handles it at folder level).
# MAGIC   - Expect ZORDER to be incremental — it rewrites the ENTIRE table
# MAGIC     (or partition) each time.  Use Liquid Clustering for incremental.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Large Delta tables (> 10 GB) with point lookup or range filter queries.
# MAGIC   - Columns with high cardinality used in WHERE: customer_id, timestamp,
# MAGIC     order_id.
# MAGIC   - NOT needed if the table is already sorted on the filter column.
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_zorder():
# MAGIC     """ZORDER BY co-locates similar values for better data skipping (Delta)."""
# MAGIC
# MAGIC     # --- Delta Lake required ---
# MAGIC     # df = spark.range(0, 1000000) \
# MAGIC     #     .withColumn("customer_id", (F.col("id") % 100000).cast("int")) \
# MAGIC     #     .withColumn("amount", (F.col("id") * 1.5).cast("double"))
# MAGIC     #
# MAGIC     # df.write.mode("overwrite").format("delta") \
# MAGIC     #     .save("/tmp/training/delta_zorder_demo")
# MAGIC     #
# MAGIC     # # --- BEFORE ZORDER: random data → wide min/max per file ---
# MAGIC     # result_before = spark.read.format("delta") \
# MAGIC     #     .load("/tmp/training/delta_zorder_demo") \
# MAGIC     #     .filter("customer_id = 500")
# MAGIC     # # All files are read — data skipping doesn't help
# MAGIC     #
# MAGIC     # # --- ZORDER BY customer_id ---
# MAGIC     # spark.sql("""
# MAGIC     #     OPTIMIZE delta.`/tmp/training/delta_zorder_demo`
# MAGIC     #     ZORDER BY (customer_id)
# MAGIC     # """)
# MAGIC     #
# MAGIC     # # --- AFTER ZORDER: co-located data → tight min/max → skip most files ---
# MAGIC     # result_after = spark.read.format("delta") \
# MAGIC     #     .load("/tmp/training/delta_zorder_demo") \
# MAGIC     #     .filter("customer_id = 500")
# MAGIC     # # Only 1-2 files read instead of all
# MAGIC
# MAGIC     print("=== ZORDER BY (Delta Lake) ===")
# MAGIC     print("Reorganizes data so similar values are in the same files")
# MAGIC     print("")
# MAGIC     print("Usage:")
# MAGIC     print("  OPTIMIZE my_table ZORDER BY (customer_id)")
# MAGIC     print("  OPTIMIZE my_table ZORDER BY (customer_id, order_date)  -- multi-column")
# MAGIC     print("")
# MAGIC     print("Effect on data skipping:")
# MAGIC     print("  Before: File min/max = [1, 99999] → every query reads every file")
# MAGIC     print("  After:  File min/max = [1, 33000] → query skips 2 out of 3 files")
# MAGIC     print("")
# MAGIC     print("Best for: high-cardinality columns in WHERE (customer_id, timestamp)")
# MAGIC     print("Limit to: 1-3 columns (effectiveness drops after that)")
# MAGIC     print("Replaces: manual sorting before write")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **8. V-ORDER (Microsoft Fabric)**
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC Standard Parquet files use generic encoding (dictionary, run-length, etc.)
# MAGIC that is good but not optimal for read speed.  In Microsoft Fabric's
# MAGIC Lakehouse, reads go through the Fabric engine which can benefit from
# MAGIC specialized encoding.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC V-Order is a WRITE-TIME optimization specific to Microsoft Fabric that
# MAGIC applies special sorting and encoding to Parquet files for faster reads.
# MAGIC
# MAGIC What it does:
# MAGIC   1. Sorts data within each row group for better compression
# MAGIC   2. Applies optimized page-level encoding
# MAGIC   3. Creates Parquet files that are 10-50% faster to read in Fabric
# MAGIC
# MAGIC The files are still standard Parquet — any engine can read them.  But
# MAGIC Fabric's reader extracts extra speed from the V-Order encoding.
# MAGIC
# MAGIC V-Order is AUTOMATIC in Fabric notebooks and pipelines.  In open-source
# MAGIC Spark, enable it manually.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Automatic in Fabric.  Manual in Spark:
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Let Fabric apply V-Order automatically (default in Fabric notebooks).
# MAGIC   - Enable manually in Spark if writing to OneLake:
# MAGIC       .option("parquet.vorder.enabled", "true")
# MAGIC   - Use for any table read frequently in Fabric — free read speedup.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Expect V-Order benefits outside Fabric — standard Parquet readers
# MAGIC     won't see the 10-50% speedup (they'll read normally).
# MAGIC   - Use V-Order as a replacement for OPTIMIZE/ZORDER — V-Order is about
# MAGIC     encoding, not file compaction or data co-location.
# MAGIC   - Worry about compatibility — V-Order files are standard Parquet;
# MAGIC     any engine (Databricks, open-source Spark) can read them normally.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Writing data to Fabric OneLake that will be read by Fabric engines.
# MAGIC   - NOT needed if data is never read through Fabric.
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_vorder():
# MAGIC     """V-Order applies special Parquet encoding for faster Fabric reads."""
# MAGIC
# MAGIC     # --- V-Order in Spark ---
# MAGIC     # df = spark.range(0, 100000) \
# MAGIC     #     .withColumn("value", (F.col("id") * 2.0).cast("double"))
# MAGIC     #
# MAGIC     # # Write with V-Order enabled
# MAGIC     # df.write.mode("overwrite") \
# MAGIC     #     .format("delta") \
# MAGIC     #     .option("parquet.vorder.enabled", "true") \
# MAGIC     #     .save("/tmp/training/vorder_demo")
# MAGIC
# MAGIC     print("=== V-Order (Microsoft Fabric) ===")
# MAGIC     print("Automatic in Fabric notebooks — no code change needed")
# MAGIC     print("")
# MAGIC     print("Manual in Spark:")
# MAGIC     print("  df.write.format('delta')")
# MAGIC     print("    .option('parquet.vorder.enabled', 'true')")
# MAGIC     print("    .save(path)")
# MAGIC     print("")
# MAGIC     print("Effect: 10-50% faster reads in Fabric engine")
# MAGIC     print("Files remain standard Parquet — any engine can read them")
# MAGIC     print("Combines with: OPTIMIZE, ZORDER, partitionBy (all orthogonal)")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **9. VACUUM (Delta Lake)**
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC Delta Lake operations create NEW files but NEVER delete old ones:
# MAGIC
# MAGIC   - OPTIMIZE rewrites small files into large ones → old small files remain
# MAGIC   - UPDATE/DELETE/MERGE use copy-on-write → old versions remain
# MAGIC   - Schema changes → old files remain
# MAGIC
# MAGIC Over time, storage grows unbounded:
# MAGIC
# MAGIC     Actual data:  10 GB
# MAGIC     Old files:    50 GB (from 30 days of OPTIMIZE, UPDATE, DELETE)
# MAGIC     Total:        60 GB → paying for 50 GB of garbage
# MAGIC
# MAGIC Old files are kept for TIME TRAVEL (read historical versions).  But after
# MAGIC the retention period, they're pure waste.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC VACUUM removes files that are:
# MAGIC   1. No longer referenced by the current Delta log
# MAGIC   2. Older than the retention period (default: 7 days / 168 hours)
# MAGIC
# MAGIC     VACUUM my_table RETAIN 168 HOURS
# MAGIC
# MAGIC After VACUUM, you CANNOT time-travel to versions older than the
# MAGIC retention period — those files are permanently deleted.
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Run VACUUM periodically after OPTIMIZE.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Run VACUUM regularly (weekly or after OPTIMIZE).
# MAGIC   - Set retention based on your time-travel needs:
# MAGIC       168 hours (7 days) = default and usually sufficient
# MAGIC   - Check storage usage before and after:
# MAGIC       DESCRIBE DETAIL my_table → sizeInBytes
# MAGIC   - Automate: schedule VACUUM as part of your maintenance pipeline.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Set retention to 0 hours — breaks concurrent readers/writers.
# MAGIC     Minimum safe: 7 days (168 hours).
# MAGIC   - VACUUM without understanding time-travel impact — once vacuumed,
# MAGIC     old versions are GONE permanently.
# MAGIC   - Run VACUUM during active long-running queries — they may reference
# MAGIC     old files that get deleted mid-query.
# MAGIC   - Skip VACUUM thinking Delta handles it — Delta NEVER auto-deletes
# MAGIC     old files.  VACUUM is the ONLY way.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - After running OPTIMIZE (old uncompacted files need cleanup).
# MAGIC   - On tables with frequent UPDATE/DELETE/MERGE operations.
# MAGIC   - When cloud storage costs are a concern.
# MAGIC   - NOT needed on immutable/append-only tables with no OPTIMIZE.
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_vacuum():
# MAGIC     """VACUUM removes old files no longer referenced by Delta log."""
# MAGIC
# MAGIC     # --- Delta Lake required ---
# MAGIC     # # Create and modify a table
# MAGIC     # df = spark.range(0, 100000) \
# MAGIC     #     .withColumn("value", (F.col("id") * 1.5).cast("double"))
# MAGIC     # df.write.mode("overwrite").format("delta").save("/tmp/training/delta_vacuum_demo")
# MAGIC     #
# MAGIC     # # Simulate updates (creates old file versions)
# MAGIC     # spark.sql("""
# MAGIC     #     UPDATE delta.`/tmp/training/delta_vacuum_demo`
# MAGIC     #     SET value = value * 2
# MAGIC     #     WHERE id < 1000
# MAGIC     # """)
# MAGIC     #
# MAGIC     # # Run OPTIMIZE (creates new compacted files, old remain)
# MAGIC     # spark.sql("OPTIMIZE delta.`/tmp/training/delta_vacuum_demo`")
# MAGIC     #
# MAGIC     # # Check size BEFORE vacuum
# MAGIC     # spark.sql("DESCRIBE DETAIL delta.`/tmp/training/delta_vacuum_demo`").select("numFiles", "sizeInBytes").show()
# MAGIC     #
# MAGIC     # # --- VACUUM: remove old files ---
# MAGIC     # spark.sql("VACUUM delta.`/tmp/training/delta_vacuum_demo` RETAIN 168 HOURS")
# MAGIC     #
# MAGIC     # # Check size AFTER vacuum
# MAGIC     # spark.sql("DESCRIBE DETAIL delta.`/tmp/training/delta_vacuum_demo`").select("numFiles", "sizeInBytes").show()
# MAGIC
# MAGIC     print("=== VACUUM (Delta Lake) ===")
# MAGIC     print("Removes old files no longer referenced by the current version")
# MAGIC     print("")
# MAGIC     print("Usage:")
# MAGIC     print("  VACUUM my_table                    -- default 7-day retention")
# MAGIC     print("  VACUUM my_table RETAIN 168 HOURS   -- explicit 7-day retention")
# MAGIC     print("  VACUUM my_table RETAIN 720 HOURS   -- 30-day retention")
# MAGIC     print("")
# MAGIC     print("WARNING: After VACUUM, time-travel to versions older than")
# MAGIC     print("         the retention period is PERMANENTLY impossible")
# MAGIC     print("")
# MAGIC     print("Maintenance pipeline order:")
# MAGIC     print("  1. Run your ETL (writes/appends/merges)")
# MAGIC     print("  2. OPTIMIZE my_table ZORDER BY (col)  -- compact + reorder")
# MAGIC     print("  3. VACUUM my_table RETAIN 168 HOURS   -- clean up old files")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # MEMORY Optimizations
# MAGIC Control how Spark uses executor/driver memory.
# MAGIC
# MAGIC | #  | Optimization              | What It Fixes                                         |
# MAGIC |----|---------------------------|-------------------------------------------------------|
# MAGIC | 1 | Caching (`df.cache()`)    | Avoids recomputation of reused DataFrames             |
# MAGIC | 2 | Persistence Levels        | Controls memory vs disk tradeoff for cached data      |
# MAGIC | 3 | `unpersist()`             | Frees memory when cached data is no longer needed     |

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **1. Caching (df.cache())**
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC A DataFrame is used multiple times in the same pipeline:
# MAGIC
# MAGIC     df = spark.read.parquet("/data/sales")        # read from disk
# MAGIC     total = df.agg(F.sum("amount")).collect()      # reads from disk
# MAGIC     by_region = df.groupBy("region").count()       # reads from disk AGAIN
# MAGIC     filtered = df.filter("amount > 1000").count()  # reads from disk AGAIN
# MAGIC
# MAGIC Each action triggers a FULL re-read and re-computation from scratch.
# MAGIC If reading from cloud storage (S3, ADLS, GCS), each re-read means:
# MAGIC   - Network I/O to fetch data again
# MAGIC   - Decompression of Parquet files again
# MAGIC   - Applying all transformations again
# MAGIC
# MAGIC For a 10 GB table used 5 times: 50 GB of I/O instead of 10 GB.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC cache() stores the DataFrame's computed result IN MEMORY on the
# MAGIC executors.  After the first action materializes the data, subsequent
# MAGIC actions read from memory instead of re-reading from disk/cloud.
# MAGIC
# MAGIC     df.cache()           # marks for caching (lazy — nothing happens yet)
# MAGIC     df.count()           # FIRST action: reads from disk, stores in memory
# MAGIC     df.filter(...).show() # SECOND action: reads from MEMORY (fast!)
# MAGIC     df.groupBy(...).count() # THIRD action: reads from MEMORY (fast!)
# MAGIC
# MAGIC cache() = persist(StorageLevel.MEMORY_AND_DISK)
# MAGIC   - Tries to store in memory
# MAGIC   - If memory is full, spills remaining partitions to disk
# MAGIC   - Data is stored DESERIALIZED (as Java/Python objects) for fast access
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Call cache() on DataFrames that are reused in multiple actions.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Cache DataFrames used in 2+ actions within the same job/notebook.
# MAGIC   - Call an action (count, show, collect) after cache() to materialize it:
# MAGIC       df.cache()
# MAGIC       df.count()  # triggers actual caching
# MAGIC   - Monitor cache usage in Spark UI → Storage tab.
# MAGIC   - unpersist() when done (see section 3) to free memory.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Cache DataFrames used only ONCE — caching adds overhead with no benefit.
# MAGIC   - Cache very large DataFrames that exceed executor memory — causes
# MAGIC     excessive spill to disk, GC pressure, and potential OOM errors.
# MAGIC   - Cache raw/unfiltered DataFrames — cache AFTER filtering to reduce
# MAGIC     the amount of data stored.
# MAGIC   - Assume cache() is immediate — it's LAZY.  Data is cached only after
# MAGIC     the first action.
# MAGIC   - Cache inside a loop that creates new DataFrames each iteration —
# MAGIC     memory fills up with stale cached data.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - DataFrame reused in multiple actions (aggregations, joins, writes).
# MAGIC   - Iterative ML algorithms that scan the same data repeatedly.
# MAGIC   - Interactive exploration in notebooks (filter, plot, re-filter).
# MAGIC   - AFTER expensive transformations (joins, aggregations) to avoid
# MAGIC     recomputing them.
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_caching():
# MAGIC     """cache() stores DataFrame in memory for reuse across actions."""
# MAGIC
# MAGIC     # Create a sample DataFrame
# MAGIC     df = spark.range(0, 1000000) \
# MAGIC         .withColumn("region", (F.col("id") % 5).cast("string")) \
# MAGIC         .withColumn("amount", (F.col("id") * 1.5).cast("double"))
# MAGIC
# MAGIC     # --- BAD: Without cache — recomputes from scratch each time ---
# MAGIC     print("=== Without cache() ===")
# MAGIC     import time
# MAGIC
# MAGIC     start = time.time()
# MAGIC     count1 = df.filter("amount > 500000").count()
# MAGIC     time1 = time.time() - start
# MAGIC
# MAGIC     start = time.time()
# MAGIC     count2 = df.groupBy("region").agg(F.sum("amount")).collect()
# MAGIC     time2 = time.time() - start
# MAGIC
# MAGIC     start = time.time()
# MAGIC     count3 = df.filter("region = '1'").count()
# MAGIC     time3 = time.time() - start
# MAGIC
# MAGIC     print(f"  Action 1: {time1:.3f}s")
# MAGIC     print(f"  Action 2: {time2:.3f}s")
# MAGIC     print(f"  Action 3: {time3:.3f}s")
# MAGIC
# MAGIC     # --- GOOD: With cache — first action caches, rest read from memory ---
# MAGIC     print("\n=== With cache() ===")
# MAGIC     df_cached = df.cache()
# MAGIC
# MAGIC     # First action materializes the cache
# MAGIC     start = time.time()
# MAGIC     _ = df_cached.count()  # triggers caching
# MAGIC     cache_time = time.time() - start
# MAGIC     print(f"  Cache materialization: {cache_time:.3f}s")
# MAGIC
# MAGIC     start = time.time()
# MAGIC     _ = df_cached.filter("amount > 500000").count()
# MAGIC     time1 = time.time() - start
# MAGIC
# MAGIC     start = time.time()
# MAGIC     _ = df_cached.groupBy("region").agg(F.sum("amount")).collect()
# MAGIC     time2 = time.time() - start
# MAGIC
# MAGIC     start = time.time()
# MAGIC     _ = df_cached.filter("region = '1'").count()
# MAGIC     time3 = time.time() - start
# MAGIC
# MAGIC     print(f"  Action 1 (from cache): {time1:.3f}s")
# MAGIC     print(f"  Action 2 (from cache): {time2:.3f}s")
# MAGIC     print(f"  Action 3 (from cache): {time3:.3f}s")
# MAGIC
# MAGIC     # Check cache in Spark UI → Storage tab
# MAGIC     print(f"\n  Is cached: {df_cached.is_cached}")
# MAGIC
# MAGIC     # Clean up
# MAGIC     df_cached.unpersist()
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2. Persistence Levels (persist with StorageLevel)
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC cache() uses MEMORY_AND_DISK by default — stores deserialized objects
# MAGIC in memory, spills to disk if full.  But this isn't always optimal:
# MAGIC
# MAGIC   - Large DataFrames: memory fills up fast → excessive GC, OOM
# MAGIC   - Memory-constrained clusters: not enough RAM for deserialized objects
# MAGIC   - Network-heavy environments: replicated caching wastes bandwidth
# MAGIC
# MAGIC You need finer control over WHERE and HOW data is cached.
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC persist(StorageLevel) gives you control over:
# MAGIC
# MAGIC   1. MEMORY vs DISK — where to store
# MAGIC   2. SERIALIZED vs DESERIALIZED — how to store
# MAGIC   3. REPLICATED vs NON-REPLICATED — how many copies
# MAGIC
# MAGIC Available StorageLevels:
# MAGIC
# MAGIC | StorageLevel              | Memory | Disk | Serialized | Replicas |
# MAGIC |---------------------------|:------:|:----:|:----------:|:--------:|
# MAGIC | MEMORY_ONLY               |   ✓    |      |            |    1     |
# MAGIC | MEMORY_AND_DISK (default) |   ✓    |  ✓   |            |    1     |
# MAGIC | MEMORY_ONLY_SER           |   ✓    |      |     ✓      |    1     |
# MAGIC | MEMORY_AND_DISK_SER       |   ✓    |  ✓   |     ✓      |    1     |
# MAGIC | DISK_ONLY                 |        |  ✓   |     ✓      |    1     |
# MAGIC | MEMORY_ONLY_2             |   ✓    |      |            |    2     |
# MAGIC | MEMORY_AND_DISK_2         |   ✓    |  ✓   |            |    2     |
# MAGIC | OFF_HEAP                  |   ✓*   |      |     ✓      |    1     |
# MAGIC
# MAGIC *OFF_HEAP stores outside the JVM heap (avoids GC pressure).*
# MAGIC
# MAGIC Key tradeoffs:
# MAGIC   - DESERIALIZED (default): faster access, more memory per object
# MAGIC   - SERIALIZED: slower access (decode needed), ~2-5x less memory usage
# MAGIC   - DISK_ONLY: slowest reads, but no memory pressure at all
# MAGIC   - REPLICATED (_2): fault-tolerant but doubles memory/disk usage
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Choose the right StorageLevel based on memory pressure and access pattern.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - Start with cache() (MEMORY_AND_DISK) — good default for most cases.
# MAGIC   - Use MEMORY_ONLY_SER when memory is tight but data is reused often:
# MAGIC       df.persist(StorageLevel.MEMORY_ONLY_SER)
# MAGIC     Saves 2-5x memory at cost of ~10-20% slower access (serialization).
# MAGIC   - Use DISK_ONLY for very large DataFrames reused a few times:
# MAGIC       df.persist(StorageLevel.DISK_ONLY)
# MAGIC     No memory pressure, but reads are slower (disk I/O).
# MAGIC   - Use OFF_HEAP for critical caches that must avoid GC pauses:
# MAGIC       df.persist(StorageLevel.OFF_HEAP)
# MAGIC     Requires: spark.memory.offHeap.enabled=true, spark.memory.offHeap.size
# MAGIC
# MAGIC DON'T:
# MAGIC   - Use MEMORY_ONLY on data larger than available executor memory —
# MAGIC     partitions that don't fit are RECOMPUTED on every access (no spill).
# MAGIC   - Use _2 (replicated) unless you're on an unstable cluster with
# MAGIC     frequent executor failures — doubles resource usage.
# MAGIC   - Mix persist levels on the same DataFrame — unpersist first, then
# MAGIC     re-persist with the new level.
# MAGIC   - Forget that persist() is also LAZY — call an action to materialize.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   MEMORY_AND_DISK (cache):  Default — most use cases.
# MAGIC   MEMORY_ONLY_SER:          Memory-constrained + frequent reuse.
# MAGIC   DISK_ONLY:                Very large data, moderate reuse.
# MAGIC   OFF_HEAP:                 Low-latency requirements, avoid GC pauses.
# MAGIC   _2 (replicated):          Unstable clusters, can't afford recomputation.
# MAGIC
# MAGIC **Implementation**
# MAGIC
# MAGIC ```
# MAGIC def demo_persistence_levels():
# MAGIC     """persist(StorageLevel) gives fine-grained control over caching."""
# MAGIC
# MAGIC     df = spark.range(0, 500000) \
# MAGIC         .withColumn("category", (F.col("id") % 10).cast("string")) \
# MAGIC         .withColumn("value", (F.col("id") * 2.5).cast("double"))
# MAGIC
# MAGIC     # --- MEMORY_ONLY: fastest reads, partitions lost if memory full ---
# MAGIC     print("=== MEMORY_ONLY ===")
# MAGIC     df_mem = df.persist(StorageLevel.MEMORY_ONLY)
# MAGIC     df_mem.count()  # materialize
# MAGIC     print(f"  Cached: {df_mem.is_cached}")
# MAGIC     print("  ✓ Fastest reads — deserialized in-memory objects")
# MAGIC     print("  ✗ No spill to disk — partitions recomputed if evicted")
# MAGIC     df_mem.unpersist()
# MAGIC
# MAGIC     # --- MEMORY_ONLY_SER: smaller memory footprint ---
# MAGIC     print("\n=== MEMORY_ONLY_SER ===")
# MAGIC     df_ser = df.persist(StorageLevel.MEMORY_ONLY_SER)
# MAGIC     df_ser.count()
# MAGIC     print(f"  Cached: {df_ser.is_cached}")
# MAGIC     print("  ✓ 2-5x less memory than MEMORY_ONLY")
# MAGIC     print("  ✗ ~10-20% slower reads (deserialization cost)")
# MAGIC     print("  Best for: memory-constrained clusters")
# MAGIC     df_ser.unpersist()
# MAGIC
# MAGIC     # --- MEMORY_AND_DISK: safe default ---
# MAGIC     print("\n=== MEMORY_AND_DISK (same as cache()) ===")
# MAGIC     df_md = df.persist(StorageLevel.MEMORY_AND_DISK)
# MAGIC     df_md.count()
# MAGIC     print(f"  Cached: {df_md.is_cached}")
# MAGIC     print("  ✓ Best of both — memory first, spill to disk")
# MAGIC     print("  ✓ No recomputation — everything is stored somewhere")
# MAGIC     print("  Default and recommended for most cases")
# MAGIC     df_md.unpersist()
# MAGIC
# MAGIC     # --- DISK_ONLY: no memory usage ---
# MAGIC     print("\n=== DISK_ONLY ===")
# MAGIC     df_disk = df.persist(StorageLevel.DISK_ONLY)
# MAGIC     df_disk.count()
# MAGIC     print(f"  Cached: {df_disk.is_cached}")
# MAGIC     print("  ✓ Zero memory pressure")
# MAGIC     print("  ✗ Slower reads (disk I/O for every access)")
# MAGIC     print("  Best for: very large DataFrames, moderate reuse")
# MAGIC     df_disk.unpersist()
# MAGIC
# MAGIC     # --- MEMORY_AND_DISK_SER: balanced ---
# MAGIC     print("\n=== MEMORY_AND_DISK_SER ===")
# MAGIC     df_mds = df.persist(StorageLevel.MEMORY_AND_DISK_SER)
# MAGIC     df_mds.count()
# MAGIC     print(f"  Cached: {df_mds.is_cached}")
# MAGIC     print("  ✓ Serialized in memory (smaller footprint)")
# MAGIC     print("  ✓ Spills to disk if memory full")
# MAGIC     print("  Best for: large data + memory-constrained + frequent reuse")
# MAGIC     df_mds.unpersist()
# MAGIC
# MAGIC     # --- Summary table ---
# MAGIC     print("\n=== Persistence Level Decision Guide ===")
# MAGIC     print("┌─────────────────────────┬─────────────┬───────────┬──────────────┐")
# MAGIC     print("│ Level                   │ Memory Cost │ Read Speed│ Best For     │")
# MAGIC     print("├─────────────────────────┼─────────────┼───────────┼──────────────┤")
# MAGIC     print("│ MEMORY_ONLY             │ High        │ Fastest   │ Small data   │")
# MAGIC     print("│ MEMORY_AND_DISK         │ High        │ Fast      │ Default      │")
# MAGIC     print("│ MEMORY_ONLY_SER         │ Low         │ Fast      │ Large + reuse│")
# MAGIC     print("│ MEMORY_AND_DISK_SER     │ Low         │ Fast      │ Large + safe │")
# MAGIC     print("│ DISK_ONLY               │ Zero        │ Slow      │ Very large   │")
# MAGIC     print("└─────────────────────────┴─────────────┴───────────┴──────────────┘")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **3. Unpersist (Freeing Cached Data)**
# MAGIC
# MAGIC **Problem**
# MAGIC
# MAGIC Every cache() call consumes executor memory.  If you cache multiple DataFrames and forget to free them, memory fills up:
# MAGIC
# MAGIC     df1.cache(); df1.count()   # 2 GB cached
# MAGIC     df2.cache(); df2.count()   # 3 GB cached
# MAGIC     df3.cache(); df3.count()   # 4 GB cached — total: 9 GB in cache
# MAGIC
# MAGIC     # df1 is no longer needed, but still consuming 2 GB
# MAGIC     # New computations get less memory → spill to disk → slower
# MAGIC
# MAGIC Worse: Spark's storage memory and execution memory SHARE the same pool.
# MAGIC Cached data that isn't freed steals memory from shuffles, sorts, and
# MAGIC aggregations — making active computations slower.
# MAGIC
# MAGIC     ┌──────────────────────────────────────────────────┐
# MAGIC     │              Spark Unified Memory Pool            │
# MAGIC     │                                                  │
# MAGIC     │  ┌──────────────┐  ┌──────────────────────────┐  │
# MAGIC     │  │  Storage     │  │   Execution              │  │
# MAGIC     │  │  (cached DFs)│  │   (shuffles, sorts,      │  │
# MAGIC     │  │  2 GB used   │←→│    aggregations)          │  │
# MAGIC     │  │              │  │   Gets less if storage    │  │
# MAGIC     │  │              │  │   is full                 │  │
# MAGIC     │  └──────────────┘  └──────────────────────────┘  │
# MAGIC     └──────────────────────────────────────────────────┘
# MAGIC
# MAGIC **Concept**
# MAGIC
# MAGIC unpersist() removes a DataFrame from the cache, freeing the memory
# MAGIC for other computations.
# MAGIC
# MAGIC     df.unpersist()           # non-blocking — marks for removal
# MAGIC     df.unpersist(blocking=True)  # blocks until fully removed
# MAGIC
# MAGIC After unpersist(), subsequent actions on the DataFrame will recompute
# MAGIC it from scratch (re-read from source, re-apply transformations).
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Always unpersist DataFrames when they're no longer needed.
# MAGIC
# MAGIC **Key Points**
# MAGIC
# MAGIC DO:
# MAGIC   - unpersist as soon as you're done with a cached DataFrame:
# MAGIC       df.cache()
# MAGIC       df.count()           # use it
# MAGIC       result = df.groupBy("col").count()  # use it
# MAGIC       df.unpersist()       # done — free the memory
# MAGIC
# MAGIC   - Use try/finally to ensure unpersist even on errors:
# MAGIC       df.cache()
# MAGIC       try:
# MAGIC           process(df)
# MAGIC       finally:
# MAGIC           df.unpersist()
# MAGIC
# MAGIC   - Check what's cached: spark.catalog.clearCache() clears everything.
# MAGIC   - Monitor Spark UI → Storage tab to see cached DataFrames and sizes.
# MAGIC
# MAGIC DON'T:
# MAGIC   - Leave DataFrames cached across notebook cells "just in case" —
# MAGIC     you'll run out of memory eventually.
# MAGIC   - unpersist a DataFrame that's still being used by downstream
# MAGIC     operations — they'll have to recompute from scratch.
# MAGIC   - Rely on Python garbage collection to unpersist — Spark's JVM cache
# MAGIC     is NOT freed when Python objects go out of scope.
# MAGIC   - Call unpersist() on a DataFrame that was never cached — it's a no-op
# MAGIC     but indicates confused logic.
# MAGIC
# MAGIC WHEN TO USE:
# MAGIC   - Immediately after the last action that uses the cached DataFrame.
# MAGIC   - At the end of a processing stage before starting the next one.
# MAGIC   - In cleanup blocks (finally) for error-safe memory management.
# MAGIC   - When switching between large cached DataFrames in a notebook.
# MAGIC """
# MAGIC
# MAGIC **Implementation**
# MAGIC ```
# MAGIC def demo_unpersist():
# MAGIC     """unpersist() frees memory when cached data is no longer needed."""
# MAGIC
# MAGIC     df1 = spark.range(0, 500000) \
# MAGIC         .withColumn("value", (F.col("id") * 1.5).cast("double"))
# MAGIC
# MAGIC     df2 = spark.range(0, 500000) \
# MAGIC         .withColumn("value", (F.col("id") * 2.5).cast("double"))
# MAGIC
# MAGIC     # --- Cache both DataFrames ---
# MAGIC     df1.cache()
# MAGIC     df2.cache()
# MAGIC
# MAGIC     # Materialize caches
# MAGIC     df1.count()
# MAGIC     df2.count()
# MAGIC
# MAGIC     print("=== Before unpersist ===")
# MAGIC     print(f"  df1 cached: {df1.is_cached}")
# MAGIC     print(f"  df2 cached: {df2.is_cached}")
# MAGIC
# MAGIC     # --- Process df1, then free it ---
# MAGIC     total1 = df1.agg(F.sum("value")).collect()[0][0]
# MAGIC     print(f"\n  df1 sum: {total1}")
# MAGIC
# MAGIC     df1.unpersist()  # done with df1 — free memory
# MAGIC     print(f"\n=== After df1.unpersist() ===")
# MAGIC     print(f"  df1 cached: {df1.is_cached}")  # False
# MAGIC     print(f"  df2 cached: {df2.is_cached}")  # True — still available
# MAGIC
# MAGIC     # --- Continue using df2 ---
# MAGIC     total2 = df2.agg(F.sum("value")).collect()[0][0]
# MAGIC     print(f"\n  df2 sum: {total2}")
# MAGIC
# MAGIC     df2.unpersist()
# MAGIC     print(f"\n=== After df2.unpersist() ===")
# MAGIC     print(f"  df2 cached: {df2.is_cached}")  # False
# MAGIC
# MAGIC     # --- Error-safe pattern ---
# MAGIC     print("\n=== Error-safe caching pattern ===")
# MAGIC     df = spark.range(0, 100000).cache()
# MAGIC     try:
# MAGIC         df.count()
# MAGIC         result = df.agg(F.sum("id")).collect()[0][0]
# MAGIC         print(f"  Result: {result}")
# MAGIC     finally:
# MAGIC         df.unpersist()
# MAGIC         print("  Cache freed in finally block")
# MAGIC
# MAGIC     # --- Clear ALL caches ---
# MAGIC     print("\n=== Clear all caches ===")
# MAGIC     spark.catalog.clearCache()
# MAGIC     print("  spark.catalog.clearCache() — all cached DataFrames freed")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # RESOURCE Optimizations
# MAGIC Control cluster-level resource allocation.
# MAGIC
# MAGIC | #  | Optimization                     | What It Fixes                                      |
# MAGIC |----|----------------------------------|----------------------------------------------------|
# MAGIC | 1 | Dynamic Resource Allocation      | Auto-scales executors up/down based on demand      |

# COMMAND ----------

# MAGIC %md
# MAGIC # Azure Synapse Spark — Nodes, Executors & vCores
# MAGIC
# MAGIC ## Complete Guide with Examples
# MAGIC
# MAGIC **Topics Covered:**
# MAGIC vCore Calculation • Node Sizing • Dynamic Executors • Wastage Analysis • Quota Management • Best Practices
# MAGIC
# MAGIC **March 2026**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 1. Fundamentals — Nodes, Executors & vCores
# MAGIC
# MAGIC ### 1.1 What is a Node?
# MAGIC
# MAGIC A Node is a Virtual Machine (VM) provisioned by Azure Synapse. It is the physical unit of compute. Every node has a fixed number of vCores and RAM determined by the Node Size family.
# MAGIC
# MAGIC > These are **Synapse-specific** node sizes, not general Azure VM SKUs.
# MAGIC
# MAGIC | Node Size  | vCores | RAM    | Family           |
# MAGIC |------------|--------|--------|------------------|
# MAGIC | Small      | 4      | 32 GB  | Memory Optimized |
# MAGIC | Medium     | 8      | 64 GB  | Memory Optimized |
# MAGIC | Large      | 16     | 128 GB | Memory Optimized |
# MAGIC | XLarge     | 32     | 256 GB | Memory Optimized |
# MAGIC | XXLarge    | 64     | 432 GB | Memory Optimized |
# MAGIC | XXXLarge   | 128    | 864 GB | Memory Optimized |
# MAGIC
# MAGIC ### 1.2 What is an Executor?
# MAGIC
# MAGIC An Executor is a JVM process that runs on a Node and performs the actual data processing. Executors receive tasks from the Driver and process Spark partitions in parallel.
# MAGIC
# MAGIC ### 1.3 What is a vCore?
# MAGIC
# MAGIC A vCore (virtual core) is a CPU thread. Each vCore processes exactly one Spark task (partition) at a time.
# MAGIC
# MAGIC **Key Rule:** `1 vCore = 1 Task = 1 Partition processed simultaneously`
# MAGIC
# MAGIC ### 1.4 The Relationship
# MAGIC
# MAGIC ```
# MAGIC Workspace
# MAGIC   └── Spark Pool (pool-level minimum: 3 nodes)
# MAGIC         └── Node (VM)
# MAGIC               └── Executor (JVM Process)
# MAGIC                     ├── vCore 1  →  Task 1  →  Partition 1
# MAGIC                     ├── vCore 2  →  Task 2  →  Partition 2
# MAGIC                     └── vCore N  →  Task N  →  Partition N
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Dynamic Executors DISABLED — 1 Node = 1 Executor
# MAGIC
# MAGIC ### 2.1 How it Works
# MAGIC
# MAGIC When Dynamic Executors (Dynamically allocate executors) is **DISABLED**, Synapse assigns the entire node to a single executor. This is a strict one-to-one mapping.
# MAGIC
# MAGIC ```
# MAGIC Node (XXLarge: 64 vCores / 432 GB)
# MAGIC   └── Executor 1
# MAGIC         ├── vCores : 64  (full node)
# MAGIC         └── Memory : 432 GB (full node)
# MAGIC ```
# MAGIC
# MAGIC > **One-to-One Rule:** Dynamic Executors OFF → 1 Node = 1 Executor = 64 vCores = 432 GB
# MAGIC
# MAGIC ### 2.2 Example — Default Config (3 Nodes, 2 Executors)
# MAGIC
# MAGIC With the Synapse **pool-level** minimum of 3 nodes and 2 default executors:
# MAGIC
# MAGIC ```
# MAGIC Node 1  →  Driver       (orchestration, no data processing)
# MAGIC Node 2  →  Executor 1   (64 vCores, processes partitions)
# MAGIC Node 3  →  Executor 2   (64 vCores, processes partitions)
# MAGIC ```
# MAGIC
# MAGIC | Metric          | Value                     |
# MAGIC |-----------------|---------------------------|
# MAGIC | Total vCores    | 3 × 64 = 192 vCores requested |
# MAGIC | Working vCores  | 2 × 64 = 128 vCores (executors only) |
# MAGIC | Parallel Tasks  | 128 tasks at a time       |
# MAGIC
# MAGIC | Node   | Role       | vCores | Processes Data?            |
# MAGIC |--------|------------|--------|----------------------------|
# MAGIC | Node 1 | Driver     | 64     | No — orchestration only    |
# MAGIC | Node 2 | Executor 1 | 64     | Yes                        |
# MAGIC | Node 3 | Executor 2 | 64     | Yes                        |
# MAGIC
# MAGIC ### 2.3 vCore Formula — Dynamic Executors OFF
# MAGIC
# MAGIC ```
# MAGIC Total vCores = (Executor Nodes + 1 Driver Node) × vCores per Node
# MAGIC              = (Executors + 1) × 64
# MAGIC
# MAGIC Example: 5 Executors
# MAGIC   = (5 + 1) × 64
# MAGIC   = 6 × 64
# MAGIC   = 384 vCores
# MAGIC ```
# MAGIC
# MAGIC ### 2.4 Node Count for Common Executor Targets
# MAGIC
# MAGIC | Executors | Nodes Needed | Total vCores | Working vCores |
# MAGIC |-----------|-------------|--------------|----------------|
# MAGIC | 2         | 3           | 192          | 128            |
# MAGIC | 4         | 5           | 320          | 256            |
# MAGIC | 5         | 6           | 384          | 320            |
# MAGIC | 10        | 11          | 704          | 640            |
# MAGIC | 20        | 21          | 1344         | 1280           |
# MAGIC
# MAGIC ### 2.5 Autoscale Behaviour — Dynamic Executors OFF
# MAGIC
# MAGIC When Dynamic Executors is **DISABLED**, autoscale **cannot add or remove executor processes dynamically**. The number of executors is fixed at session start based on the explicit executor count (either the default or what was set via `spark.executor.instances` / `%%configure`).
# MAGIC
# MAGIC However, **the node count is still determined by how many executors are configured**. If `spark.executor.instances` is set to a high value (e.g., 10), Synapse will provision 11 nodes (10 executors + 1 driver) at session start and hold them for the entire session — it will **not scale down** during idle periods or **scale up** mid-session in response to workload.
# MAGIC
# MAGIC ```
# MAGIC Autoscale: Min 3, Max 20 — Dynamic Executors OFF
# MAGIC
# MAGIC Session starts with spark.executor.instances = 2
# MAGIC   → 3 nodes provisioned (min)
# MAGIC   → stays at 3 nodes for entire session
# MAGIC   → Max Nodes = 20 is NEVER reached via autoscale
# MAGIC
# MAGIC Session starts with spark.executor.instances = 10
# MAGIC   → 11 nodes provisioned at start (10 executors + 1 driver)
# MAGIC   → stays at 11 nodes for entire session
# MAGIC   → autoscale does NOT scale this down during idle
# MAGIC ```
# MAGIC
# MAGIC > **Key Point:** With Dynamic Executors OFF, the node count is locked for the session. Autoscale's Min/Max range only acts as a ceiling — it does **not** dynamically adjust nodes mid-session. True elastic scaling requires Dynamic Executors ON.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Dynamic Executors ENABLED — Multiple Executors per Node
# MAGIC
# MAGIC ### 3.1 How it Works
# MAGIC
# MAGIC When Dynamic Executors is **ENABLED**, Synapse packs multiple smaller executors into each node. For XXLarge nodes, Synapse **by default** fits 4 executors per node, each receiving a quarter of the node's resources.
# MAGIC
# MAGIC ```
# MAGIC Node (XXLarge: 64 vCores / 432 GB) — Dynamic Executors ON
# MAGIC   ├── Executor 1  →  16 vCores / 108 GB
# MAGIC   ├── Executor 2  →  16 vCores / 108 GB
# MAGIC   ├── Executor 3  →  16 vCores / 108 GB
# MAGIC   └── Executor 4  →  16 vCores / 108 GB
# MAGIC ```
# MAGIC
# MAGIC ```
# MAGIC 1 Node = 4 Executors (default for XXLarge)
# MAGIC vCores per Executor = 64 ÷ 4 = 16
# MAGIC ```
# MAGIC
# MAGIC > **Note:** The 4-executors-per-node ratio is the Synapse default for XXLarge. The actual count may vary if `spark.executor.cores` or `spark.executor.memory` are overridden in `%%configure` or pool settings.
# MAGIC
# MAGIC **Key Difference:** Dynamic Executors ON → vCores requested = Nodes × 64, NOT Executors × 64
# MAGIC
# MAGIC ### 3.2 vCore Formula — Dynamic Executors ON
# MAGIC
# MAGIC ```
# MAGIC vCores per Executor = Node vCores ÷ Executors per Node
# MAGIC                     = 64 ÷ 4 = 16
# MAGIC
# MAGIC Total vCores = Nodes × 64  (always based on nodes, not executors)
# MAGIC
# MAGIC Example: 8 Executors
# MAGIC   Nodes needed = (8 ÷ 4) + 1 driver = 3 nodes
# MAGIC   Total vCores = 3 × 64 = 192 vCores
# MAGIC   NOT 8 × 64 = 512 (this is WRONG with Dynamic Executors ON)
# MAGIC ```
# MAGIC
# MAGIC ### 3.3 Comparison: Dynamic Executors OFF vs ON
# MAGIC
# MAGIC | Setting                  | Dyn Exec OFF          | Dyn Exec ON              |
# MAGIC |--------------------------|-----------------------|--------------------------|
# MAGIC | Mapping                  | 1 Node = 1 Executor   | 1 Node = 4 Executors*    |
# MAGIC | vCores per Executor      | 64 (full node)        | 16 (quarter node)*       |
# MAGIC | RAM per Executor         | 432 GB                | 108 GB*                  |
# MAGIC | 8 Executors → Nodes      | 9 nodes               | 3 nodes                  |
# MAGIC | 8 Executors → vCores     | 9 × 64 = 576          | 3 × 64 = 192             |
# MAGIC | Autoscale works?         | No                    | Yes                      |
# MAGIC | Fault tolerance          | Lower                 | Higher                   |
# MAGIC | Resource efficiency      | Lower                 | Higher                   |
# MAGIC
# MAGIC *\*Default for XXLarge. May vary if executor core/memory settings are customized.*
# MAGIC
# MAGIC ### 3.4 Perfect Executor Count — Fill All Nodes
# MAGIC
# MAGIC To avoid wasting vCores on partially filled nodes, always set Max Executors as a **multiple of 4** (for XXLarge nodes at default settings).
# MAGIC
# MAGIC ```
# MAGIC Min Nodes = 3 (Synapse pool-level default)
# MAGIC Executor Nodes = 3 - 1 driver = 2 nodes
# MAGIC Max Executors  = 2 × 4 = 8  ← fills all nodes perfectly
# MAGIC
# MAGIC Node 1  →  Driver
# MAGIC Node 2  →  Executor 1 + 2 + 3 + 4  (full)
# MAGIC Node 3  →  Executor 5 + 6 + 7 + 8  (full)
# MAGIC
# MAGIC Total vCores = 3 × 64 = 192
# MAGIC Wasted vCores = 0
# MAGIC ```
# MAGIC
# MAGIC | Max Executors | Nodes Used | Node 3 Usage         | vCores Wasted |
# MAGIC |---------------|-----------|----------------------|---------------|
# MAGIC | 4             | 3         | Idle — wasted        | 64 vCores     |
# MAGIC | 5             | 3         | 25% used (1 of 4)    | 48 vCores     |
# MAGIC | 6             | 3         | 50% used (2 of 4)    | 32 vCores     |
# MAGIC | 7             | 3         | 75% used (3 of 4)    | 16 vCores     |
# MAGIC | 8             | 3         | 100% full            | 0 — perfect   |
# MAGIC
# MAGIC > **Best Practice:** For XXLarge nodes (default config), always set Max Executors as a multiple of 4 (4, 8, 12, 16...) to avoid partial node wastage.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. vCore Calculation & Workspace Quota
# MAGIC
# MAGIC ### 4.1 Your Pool Config (from screenshot)
# MAGIC
# MAGIC | Setting            | Value                              |
# MAGIC |--------------------|------------------------------------|
# MAGIC | Node Size Family   | Memory Optimized                   |
# MAGIC | Node Size          | XXLarge (64 vCores / 432 GB)       |
# MAGIC | Autoscale          | Enabled (Min 3, Max 20)            |
# MAGIC | Dynamic Executors  | DISABLED                           |
# MAGIC | Intelligent Cache  | 50%                                |
# MAGIC | Workspace Quota    | 2000 vCores                        |
# MAGIC | Available vCores   | 416 vCores (from quota error)      |
# MAGIC
# MAGIC ### 4.2 Why the CAPACITY_EXCEEDED Error Occurred
# MAGIC
# MAGIC ```
# MAGIC Error: Your job requested 704 vCores
# MAGIC        Only 416 available
# MAGIC ```
# MAGIC
# MAGIC **Breakdown:**
# MAGIC
# MAGIC ```
# MAGIC 704 = 11 nodes × 64 vCores
# MAGIC     = 10 executor nodes + 1 driver
# MAGIC ```
# MAGIC
# MAGIC **Root Cause:** With Dynamic Executors OFF, `spark.executor.instances` was set to 10 (either explicitly via `%%configure`, pipeline config, or a parent notebook call). Since each executor needs its own dedicated node, Synapse provisioned 11 nodes (10 executors + 1 driver) at session start — all 704 vCores at once.
# MAGIC
# MAGIC Meanwhile, the workspace already had other pools consuming quota:
# MAGIC
# MAGIC ```
# MAGIC 2000 - 416 = 1584 vCores used by other pools/sessions
# MAGIC Only 416 remained → 704 > 416 → FAILED
# MAGIC ```
# MAGIC
# MAGIC > **Fix:** Either reduce executor count to fit within available quota, cap Max Nodes, or enable Dynamic Executors so that 10 executors fit on fewer nodes (3 nodes = 192 vCores instead of 704).
# MAGIC
# MAGIC ### 4.3 Safe Node Calculation for Your Workspace
# MAGIC
# MAGIC ```
# MAGIC Available vCores        = 416
# MAGIC vCores per Node         = 64
# MAGIC
# MAGIC Max safe nodes          = 416 ÷ 64 = 6.5 → 6 nodes
# MAGIC Max executor nodes      = 6 - 1 driver  = 5
# MAGIC Max safe executors      = 5  (Dyn OFF) or 20 (Dyn ON, 4 per node)
# MAGIC
# MAGIC Safe vCores used        = 6 × 64 = 384
# MAGIC Remaining for others    = 416 - 384 = 32 vCores headroom
# MAGIC ```
# MAGIC
# MAGIC ### 4.4 Recommended Config for Your Pool
# MAGIC
# MAGIC | Setting          | Current       | Recommended                     |
# MAGIC |------------------|---------------|---------------------------------|
# MAGIC | Min Nodes        | 3             | 3                               |
# MAGIC | Max Nodes        | 20            | 6                               |
# MAGIC | Dynamic Executors| Disabled      | **Enable** (Min 2, Max 8)       |
# MAGIC | Max Executors    | N/A           | 8 (multiple of 4)               |
# MAGIC | vCores at start  | 192           | 192 (3 nodes × 64)              |
# MAGIC | vCores at max    | 1344          | 384 (6 nodes × 64)              |
# MAGIC | Fits 416 quota?  | No            | **Yes**                         |
# MAGIC | Parallel Tasks   | 128           | 128 (same!)                     |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5. Wastage Analysis
# MAGIC
# MAGIC ### 5.1 Types of Wastage
# MAGIC
# MAGIC | Wastage Type       | Cause                                     | Avoidable? | Fix                              |
# MAGIC |--------------------|-------------------------------------------|------------|----------------------------------|
# MAGIC | Driver node        | Full node reserved for orchestration      | No         | Unavoidable in Spark             |
# MAGIC | Idle nodes         | Nodes provisioned but no executor         | Yes        | Set Max Nodes = Executors + 1    |
# MAGIC | Partial node       | Executors don't fill all node slots       | Yes        | Use multiples of 4               |
# MAGIC | Last batch         | Final batch has fewer tasks than vCores   | Mostly     | Align partition count            |
# MAGIC | Between stages     | All executors idle during shuffle         | Yes        | Enable Dynamic Executors         |
# MAGIC | Session idle       | Notebook not running but pool is ON       | Yes        | Reduce auto-pause to 5 min      |
# MAGIC | Max Nodes unused   | Max set high but never reached            | N/A        | No cost — nodes not provisioned  |
# MAGIC
# MAGIC ### 5.2 Example — Nodes=3, Executors=2, Dynamic OFF
# MAGIC
# MAGIC ```
# MAGIC Node 1  →  Driver      →  useful (orchestration)
# MAGIC Node 2  →  Executor 1  →  useful (64 vCores processing)
# MAGIC Node 3  →  Executor 2  →  useful (64 vCores processing)
# MAGIC ```
# MAGIC
# MAGIC | Metric          | Value                        |
# MAGIC |-----------------|------------------------------|
# MAGIC | Wasted nodes    | 0                            |
# MAGIC | Wasted vCores   | 0 (executor nodes fully used)|
# MAGIC | Driver overhead | 64 vCores (unavoidable)      |
# MAGIC
# MAGIC > **Note:** Max Nodes setting does NOT cause wastage. Unprovisioned nodes cost nothing. Only provisioned nodes are billed.
# MAGIC
# MAGIC ### 5.3 Example — Max Executors=5, Dynamic ON, Nodes=3
# MAGIC
# MAGIC ```
# MAGIC Node 1  →  Driver
# MAGIC Node 2  →  Executor 1 + 2 + 3 + 4  (4 slots, full)
# MAGIC Node 3  →  Executor 5               (1 of 4 slots used)
# MAGIC            Empty slot  ← 16 vCores wasted
# MAGIC            Empty slot  ← 16 vCores wasted
# MAGIC            Empty slot  ← 16 vCores wasted
# MAGIC ```
# MAGIC
# MAGIC | Metric              | Value              |
# MAGIC |---------------------|--------------------|
# MAGIC | Node 3 utilization  | 25%                |
# MAGIC | vCores wasted       | 48 out of 64       |
# MAGIC | Fix                 | Set Max Executors = 8 to fill Node 3 |
# MAGIC
# MAGIC ### 5.4 Quota Blocking vs Billing
# MAGIC
# MAGIC These are two different things that are often confused:
# MAGIC
# MAGIC | Aspect    | Quota Blocking                         | Billing                         |
# MAGIC |-----------|----------------------------------------|---------------------------------|
# MAGIC | Based on  | Nodes provisioned at session start     | Actual nodes running over time  |
# MAGIC | At start  | All provisioned nodes block quota      | All provisioned nodes billed    |
# MAGIC | Scale-up  | Additional nodes block more quota      | Additional nodes add cost       |
# MAGIC | Scale-down| Released nodes free quota              | Reduced cost                    |
# MAGIC
# MAGIC > **Nuance with Dynamic Allocation ON:** Synapse may initially reserve quota for only the min executors' nodes and acquire additional quota as it scales up. If the workspace is near capacity, scale-up may **fail mid-job** rather than failing at session start. Plan headroom accordingly.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 6. Partition & Task Planning
# MAGIC
# MAGIC ### 6.1 How Tasks Map to vCores
# MAGIC
# MAGIC ```
# MAGIC Total vCores (executors) = 2 × 64 = 128 (Dyn OFF, 2 executors)
# MAGIC Total tasks              = 2000
# MAGIC
# MAGIC Batch 1  : Tasks    1–128   →  128 vCores busy  (full)
# MAGIC Batch 2  : Tasks  129–256   →  128 vCores busy  (full)
# MAGIC ...
# MAGIC Batch 15 : Tasks 1793–1920  →  128 vCores busy  (full)
# MAGIC Batch 16 : Tasks 1921–2000  →   80 vCores busy  (partial — 48 idle!)
# MAGIC ```
# MAGIC
# MAGIC ### 6.2 Ideal Partition Count Formula
# MAGIC
# MAGIC ```
# MAGIC Rule 1 (vCore based):
# MAGIC   ideal_partitions = total_executor_vCores × 2
# MAGIC   Example: 128 vCores × 2 = 256 partitions
# MAGIC
# MAGIC Rule 2 (data size based):
# MAGIC   ideal_partitions = total_data_size_MB ÷ 128
# MAGIC   Example: 500 GB = 500,000 MB ÷ 128 = ~3906 partitions
# MAGIC
# MAGIC Use the HIGHER of both values.
# MAGIC ```
# MAGIC
# MAGIC ### 6.3 Setting Shuffle Partitions
# MAGIC
# MAGIC ```python
# MAGIC # Option 1: Set fixed partitions matched to executor vCores
# MAGIC spark.conf.set('spark.sql.shuffle.partitions', '256')
# MAGIC
# MAGIC # Option 2 (Recommended): Use AQE to auto-tune at runtime
# MAGIC spark.conf.set('spark.sql.adaptive.enabled', 'true')
# MAGIC spark.conf.set('spark.sql.adaptive.coalescePartitions.enabled', 'true')
# MAGIC # Set a high initial value — AQE will coalesce small partitions automatically
# MAGIC spark.conf.set('spark.sql.shuffle.partitions', '200')
# MAGIC ```
# MAGIC
# MAGIC > **Note:** With AQE enabled, Spark dynamically coalesces partitions at runtime. You do not need to set a precise value — AQE adjusts based on actual data sizes during execution.
# MAGIC
# MAGIC ### 6.4 Align Partitions to Avoid Last Batch Waste
# MAGIC
# MAGIC ```
# MAGIC Executor vCores = 128
# MAGIC Tasks = 2000
# MAGIC
# MAGIC 2000 ÷ 128 = 15.625  →  last batch is partial (waste!)
# MAGIC
# MAGIC Fix: Round up to nearest multiple of 128
# MAGIC   128 × 16 = 2048 partitions
# MAGIC   2048 ÷ 128 = 16 perfect batches, zero last-batch waste
# MAGIC ```
# MAGIC
# MAGIC ```python
# MAGIC spark.conf.set('spark.sql.shuffle.partitions', '2048')
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 7. Best Practices & Recommended Config
# MAGIC
# MAGIC ### 7.1 Optimal Pool Settings for Your Workspace
# MAGIC
# MAGIC | Setting           | Value       | Reason                              |
# MAGIC |-------------------|-------------|-------------------------------------|
# MAGIC | Node Size         | XXLarge     | Existing — fits use case            |
# MAGIC | Min Nodes         | 3           | Synapse pool-level minimum          |
# MAGIC | Max Nodes         | 6           | Stays within 416 vCore limit        |
# MAGIC | Dynamic Executors | **ON**      | Efficient resource use + autoscale  |
# MAGIC | Min Executors     | 2           | Minimum working config              |
# MAGIC | Max Executors     | 8           | Multiple of 4, fills 2 executor nodes |
# MAGIC | Auto-pause        | 5–10 min    | Avoid idle billing                  |
# MAGIC | Intelligent Cache | 50%         | Keep existing setting               |
# MAGIC
# MAGIC ### 7.2 Recommended Spark Config (PySpark)
# MAGIC
# MAGIC ```python
# MAGIC # At top of every notebook
# MAGIC spark.conf.set('spark.sql.adaptive.enabled',                    'true')
# MAGIC spark.conf.set('spark.sql.adaptive.coalescePartitions.enabled', 'true')
# MAGIC spark.conf.set('spark.sql.adaptive.skewJoin.enabled',           'true')
# MAGIC spark.conf.set('spark.sql.shuffle.partitions',                  '256')
# MAGIC spark.conf.set('spark.dynamicAllocation.executorIdleTimeout',   '30s')
# MAGIC ```
# MAGIC
# MAGIC ### 7.3 Golden Rules
# MAGIC
# MAGIC 1. **1 vCore = 1 Task = 1 Partition** processed at a time
# MAGIC 2. **Total vCores = Nodes × vCores per Node** (always based on nodes)
# MAGIC 3. **Dynamic Executors OFF** → 1 Node = 1 Executor (one-to-one)
# MAGIC 4. **Dynamic Executors ON** → 1 Node = 4 Executors (default for XXLarge)
# MAGIC 5. **Max Nodes = Desired Executors + 1 Driver** (when Dyn OFF)
# MAGIC 6. **Max Executors = multiple of 4** to avoid partial node waste (when Dyn ON, XXLarge default)
# MAGIC 7. **Driver node is unavoidable** — always costs 1 full node
# MAGIC 8. **Unprovisioned nodes cost nothing** — Max Nodes is just a ceiling
# MAGIC 9. **Autoscale only works when Dynamic Executors is ON**
# MAGIC 10. **Align shuffle partitions** to executor vCores × 2
# MAGIC
# MAGIC ### 7.4 Quick Decision Guide
# MAGIC
# MAGIC | Scenario                          | Action                                         |
# MAGIC |-----------------------------------|-------------------------------------------------|
# MAGIC | Getting CAPACITY_EXCEEDED         | Reduce Max Nodes or enable Dynamic Executors    |
# MAGIC | Nodes not scaling up              | Enable Dynamic Executors                        |
# MAGIC | Partial node waste                | Set Max Executors = multiple of 4               |
# MAGIC | High idle cost between stages     | Enable Dynamic Executors                        |
# MAGIC | Session idle billing              | Reduce auto-pause to 5 minutes                  |
# MAGIC | Need more parallelism             | Increase Max Executors (multiple of 4)          |
# MAGIC | Need to protect workspace quota   | Reduce Max Nodes                                |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC For your XXLarge pool with **416 available vCores**: Enable Dynamic Executors (Min 2, Max 8), set Max Nodes = 6. This gives zero partial-node wastage, 192 total vCores, 128 parallel tasks, and stays safely within quota.
# MAGIC