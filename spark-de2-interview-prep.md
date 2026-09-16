# Spark Interview Prep — Microsoft Data Engineer 2

> **How to use this file:** Each question is phrased the way an interviewer will actually say it out loud, followed by a model answer and the follow-up probes they use to go deeper. Rehearse the answers until you can say them conversationally, not from memory. The final "Closer" is the one they almost always end with — have it cold.

**The mental model that answers 80% of Spark questions:**
> Every slow Spark job is one of four things — **skew, spill, shuffle, or small files**. Diagnose from the Spark UI, then apply the *least-invasive* fix first. Fixing the data layout beats growing the cluster.

---

## Table of Contents
- [Round 1 — Internals Warm-up](#round-1--internals-warm-up)
- [Round 2 — Core Scenario Grilling](#round-2--core-scenario-grilling)
- [Round 3 — Rapid-Fire Depth Checks](#round-3--rapid-fire-depth-checks)
- [SQL & Python Quick Wins](#sql--python-quick-wins)
- [The Closer](#the-closer)
- [Cheat-Sheet Recall Table](#cheat-sheet-recall-table)

---

## Round 1 — Internals Warm-up
*They're checking if you know internals, not memorized definitions.*

### "Walk me through what happens when you hit `df.write` or `.collect()` in Spark. What's going on under the hood?"
An action triggers a **job**. Spark builds a DAG and splits it into **stages** at every shuffle boundary. Each stage becomes a set of **tasks** — one per partition — that the driver schedules onto executor cores. The driver plans, executors do the work and hold cached data.

> **Follow-up: "What exactly creates a new stage?"**
> A **wide transformation** — anything needing a shuffle: `groupBy`, `join`, `distinct`, `repartition`, window functions. Narrow ops like `map`/`filter` stay in the same stage.

### "What's the difference between a narrow and a wide transformation, and why should I care?"
Narrow = each input partition maps to one output partition, no data movement. Wide = data gets shuffled across the network — disk I/O, serialization, network cost. You care because *every* performance problem in Spark traces back to a wide transformation. Optimization is mostly about minimizing or fixing shuffles.

### "If I gave you an RDD-based job, would you rewrite it? Why?"
Yes, to DataFrames. RDDs are opaque to **Catalyst** — Spark can't optimize your closures. DataFrames get predicate pushdown, column pruning, and Tungsten whole-stage codegen for free. I'd only stay on RDD for truly custom partitioning or non-tabular logic.

### "How does Spark recover when an executor dies mid-job? It doesn't replicate data like HDFS, right?"
Correct — it uses **lineage**. Every DataFrame remembers how it was derived from its parents, so a lost partition is just recomputed from source. For long lineages — iterative or streaming — I'd **checkpoint** to truncate the lineage so recovery doesn't replay the whole chain.

### "What are Catalyst and Tungsten?"
- **Catalyst** is the query optimizer: parse → analyze → logical optimization (predicate pushdown, column pruning, constant folding) → physical planning → cost-based selection.
- **Tungsten** is the execution engine: off-heap memory, cache-aware computation, and **whole-stage code generation** that collapses operators into one optimized Java function.

---

## Round 2 — Core Scenario Grilling
*This is where DE2 is decided.*

### "We've got a Spark job processing ~300GB daily. It used to finish in 40 minutes, now it's taking 3 hours. Nobody changed the code. Where do you start?"
Work a checklist in priority order:
1. **Data growth** — did input volume or row count jump? Most common cause, check first.
2. **Skew** — Spark UI task-time distribution in the heavy stage; a long tail means one key/partition dominates.
3. **Spill** — are memory/disk spill metrics climbing? Partitions no longer fit in memory.
4. **Small files** — did input file count explode? Metadata/listing overhead alone can add hours.
5. **Plan change** — `explain()` diff. Often a table crossed the broadcast threshold and a fast broadcast join silently became a sort-merge join.

Nine times out of ten it's data growth tipping the job into skew or a plan change.

> **Follow-up: "You find one stage where 5 tasks still run and 195 finished instantly. What's that telling you?"**
> Skew. Five partitions hold almost all the data. The cluster looks idle because 195 cores wait on 5 stragglers — a **stage barrier**. Enable AQE skew join, salt the hot key, or repartition.

### "A join stage shows 200 tasks, but only ~5 run at a time and the cluster sits idle. Explain that."
Two possibilities. Most likely **skewed keys** — those 5 tasks got the fat partitions. The other is the **200** itself: that's the default `spark.sql.shuffle.partitions`, and on genuinely large data it caps parallelism while each partition is huge. Check the task-size distribution to tell them apart, then turn on AQE — `coalescePartitions` right-sizes the count and `skewJoin` splits hot partitions automatically.

### "Your aggregation runs fine on 50GB but OOMs at 1TB. You can't just ask for more memory. What do you do?"
Redesign the data flow, not the cluster:
- **Pre-aggregate** early — map-side combine, `reduceByKey` over `groupByKey`.
- **Two-stage (salted) aggregation** — partial agg on a salted key, then final agg — kills skew-driven OOM.
- **Incremental processing** — aggregate daily deltas and merge, not full 1TB recompute.
- **Narrower shuffle** — fewer columns, higher `shuffle.partitions` so each reducer handles a bounded slice.
- **Partition on the aggregation key** so no single reducer sees an unbounded group.

Memory is the last lever, not the first.

### "We write to S3 partitioned by date. Downstream is slow because of thousands of tiny files. Fix it without wrecking ingestion latency."
Small files come from too many concurrent writers per partition. So:
- Keep ingestion as low-latency **micro-batches**, but run a **background compaction** job that rewrites each partition into ~128MB–1GB files.
- Control writer output with `coalesce` before write or `spark.sql.maxRecordsPerFile`.
- If it's Delta, this is just `OPTIMIZE` (plus auto-compaction) — it merges files transactionally without blocking writes.

Ingestion stays fast, readers get healthy file sizes.

### "A broadcast join that ran fine for months suddenly fails in production. The 'small' table is still small. What happened?"
The **serialized in-memory size** blew past the broadcast threshold even though on-disk Parquet still looks tiny — compression hides the real footprint, and it materializes much larger in memory. Add growth and it tips over. Could also be **broadcast timeout** or **executor memory pressure** since every executor holds a copy. Fix: check the actual broadcast size, raise the threshold deliberately or bump executor memory, and let AQE fall back to sort-merge if it no longer qualifies.

### "CPU usage is low, but the job crawls. If it's not CPU-bound, what is it?"
Idle CPU means it's waiting on something else. Check, in order: **shuffle spill** (disk I/O), **I/O wait** from slow storage or millions of small files, **skew** (one core pinned, rest idle), **network** (shuffle fetch across nodes), and **metadata overhead** (slow S3 listing on huge file counts). Low CPU plus slow is almost always I/O or skew — never compute.

### "I doubled the cluster size and the job got *slower*. How?"
Distributed overhead outran the benefit. More executors mean more **scheduling/coordination** cost, and more shuffle partitions mean **shuffle amplification** — an N×M explosion of network fetches and connections, plus **network contention**. If the job is skewed or tasks are small, extra nodes just add coordination for no useful work. The fix is right-sizing, not up-sizing.

### "One key is causing most of the delay. How do you find it, and how do you handle it?"
Find it: Spark UI long-tail on task time, or `df.groupBy(key).count().orderBy(desc)` to confirm the hot key. Handle it:
- **Salting** — random suffix on the hot key, aggregate in two stages, strip the salt.
- **Key isolation** — route the hot key down a separate path and union results.
- **AQE skew join** to auto-split it.
- **Partial/map-side aggregation** to shrink data before it shuffles.

### "We ingest CDC continuously. Updates sometimes arrive hours late. Design it so late data doesn't corrupt aggregates."
Never blindly increment an aggregate. Instead:
- **MERGE-based upserts** keyed on primary key + version/timestamp — apply a row only if it's newer.
- A bounded **recomputation window** — reprocess the last N days each run so late arrivals get absorbed.
- **Watermarking** to define how late is too late and cap state growth.
- **Stateful streaming** with `withWatermark` for the streaming path.

The aggregate is always reconciled against the latest version, so replays and late updates are idempotent.

### "Build a pipeline that supports replay and backfill without producing duplicate results."
**Idempotency** is the whole game. `MERGE`/upsert on a business key so re-running overwrites instead of appending. Make partitions the atomic unit — reprocess and **overwrite a whole date partition** rather than inserting. Add **watermarking** for streaming and **checkpointing** for exactly-once. In Delta, `MERGE` + partition overwrite + time travel gives clean, repeatable backfills.

---

## Round 3 — Rapid-Fire Depth Checks
*Short, sharp — they're probing breadth.*

### "`repartition` vs `coalesce` — quick."
`repartition` = full shuffle, up or down, even sizes — use before wide ops or to fix skew. `coalesce` = narrow, decrease only, no shuffle — use before writing to cut small files.

### "What's `spark.sql.shuffle.partitions` and why does the default bite people?"
Default 200. Too many for small data (tiny partitions, overhead), too few for big data (huge partitions, spill/OOM). AQE's coalesce fixes it dynamically now.

### "Broadcast Hash Join vs Sort-Merge Join vs Shuffle Hash Join — when each?"
- **Broadcast Hash Join** when one side fits under ~10MB (`autoBroadcastJoinThreshold`) — no shuffle, fastest.
- **Sort-Merge Join** for two large tables — both shuffled and sorted on the key. Robust default.
- **Shuffle Hash Join** when one side fits in memory per partition and sort-merge is disabled/inefficient.
- AQE can switch sort-merge to broadcast at runtime if a side turns out small.

### "What is bucketing and why does it help joins?"
Pre-hashing tables into a fixed number of buckets on the join key at write time. Two bucketed tables joined on that key skip the shuffle entirely — co-located buckets join directly. Great for repeated joins on the same key.

### "When would you actually `cache()`, and when is it a mistake?"
Cache when a DataFrame is reused multiple times — iterative work, repeated joins. Mistake when used once: it wastes memory and evicts execution memory, slowing the job. Always `unpersist()`. Prefer `MEMORY_AND_DISK` to avoid recompute on eviction.

### "Explain Spark's memory model in 30 seconds."
Executor memory = reserved + user + unified. **Unified** memory (~60%, `spark.memory.fraction`) is shared between **execution** (shuffle/sort/join buffers) and **storage** (cache); they borrow dynamically, and execution can evict cache but not vice versa. Plus overhead and off-heap.

### "Why does a job OOM, and how do you tell driver OOM from executor OOM?"
OOM causes: skewed partitions, wide aggregations building huge hash maps, exploding joins, too-small `shuffle.partitions`. **Driver OOM** is usually `collect()`/`toPandas()`, oversized broadcast, or huge query plans. **Executor OOM** is skew, wide shuffles, or large caches. Different root cause → different fix.

### "Parquet vs Avro vs ORC — one line each."
Parquet: columnar, best for analytical reads (pruning + pushdown). ORC: columnar, strong in Hive/Presto with ACID. Avro: row-based, best for streaming ingestion and schema evolution.

### "What does AQE do?"
Re-optimizes at runtime using real shuffle stats: **coalesce shuffle partitions** (fixes the 200 default), **skew join handling** (auto-splits skewed partitions), and **dynamic join switching** (sort-merge → broadcast when a side turns out small). On by default in Spark 3.2+.

### "What does `OPTIMIZE` with Z-ORDER do in Delta?"
Compacts small files and co-locates related data so data-skipping reads far fewer files — the production fix for both small files and slow filtered queries. Pair with `MERGE` for CDC/upserts, time travel for replay, `VACUUM` for cleanup.

### "Microsoft-specific — how does this map to Fabric / Synapse?"
Microsoft **Fabric** runs Spark on **OneLake** (Delta/Parquet underneath); Synapse Spark pools for ETL. Every optimization principle here — skew, shuffle, small files, `OPTIMIZE` — transfers directly.

---

## SQL & Python Quick Wins

### "Remove duplicate records while keeping the latest entry."
```sql
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY business_key ORDER BY updated_at DESC
  ) AS rn
  FROM events
)
SELECT * FROM ranked WHERE rn = 1;
```

### "Find the second highest salary."
```sql
SELECT MAX(salary) FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Robust to ties / N-th highest:
SELECT DISTINCT salary FROM employees
ORDER BY salary DESC OFFSET 1 ROWS FETCH NEXT 1 ROWS ONLY;
```

### "What are window functions and when do you use them?"
Aggregation **without collapsing rows**: running totals, rankings (`ROW_NUMBER`/`RANK`/`DENSE_RANK`), dedup-keep-latest, lag/lead comparisons, per-group top-N.

### "Write a function to find the frequency of elements in a dataset."
```python
from collections import Counter

def frequency(data):
    return dict(Counter(data))

# PySpark equivalent: df.groupBy("col").count()
```

---

## The Closer
*They almost always end with this. Have it cold.*

### "At a senior level, Spark expertise comes down to one thing: why is this job slow, and what's the most impactful fix?"
Every slow Spark job is one of four things — **skew, spill, shuffle, or small files**. I diagnose from the Spark UI: stage duration, task-time distribution for skew, spill metrics for memory pressure, shuffle read/write for network, and input file count for metadata overhead. Then I apply the *least-invasive* fix first — AQE and skew handling, then repartition or coalesce, then broadcast, then compaction — and only add memory or nodes as a last resort, because over-scaling just adds coordination and shuffle overhead. The most impactful fix is almost always fixing the data layout, not growing the cluster.

---

## Cheat-Sheet Recall Table

| Symptom | Most likely cause | First fix |
|---|---|---|
| Runtime suddenly doubled/tripled | Data growth → skew or plan change | `explain()` diff + Spark UI task distribution |
| Few tasks run, cluster idle | Skewed keys / too few partitions | AQE skew join + salting |
| OOM at scale | Fat partitions / wide agg | Pre-aggregate, salt, incremental, narrower shuffle |
| Downstream slow, tiny files | Too many writers per partition | Compaction / `OPTIMIZE`, `coalesce` on write |
| Broadcast join fails after growth | Serialized size > threshold | Check real broadcast size, raise threshold / fall back to SMJ |
| Low CPU but slow | I/O wait / spill / skew | Fix shuffle spill + file layout |
| Bigger cluster = slower | Coordination + shuffle amplification | Right-size, don't up-size |
| Single hot key | Data skew | Salting / key isolation / AQE |
| Late CDC corrupts aggregates | Non-idempotent writes | MERGE upsert + recomputation window + watermark |
| Backfill duplicates results | Non-idempotent pipeline | MERGE + partition overwrite + time travel |

| Config | Default | What it controls |
|---|---|---|
| `spark.sql.shuffle.partitions` | 200 | Partitions after a shuffle |
| `spark.sql.autoBroadcastJoinThreshold` | 10MB | Broadcast join cutoff |
| `spark.sql.adaptive.enabled` | true (3.2+) | AQE master switch |
| `spark.memory.fraction` | ~0.6 | Unified execution+storage memory |
| `spark.sql.broadcastTimeout` | 300s | Broadcast wait limit |
