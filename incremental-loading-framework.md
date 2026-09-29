# Incremental Loading Framework in Microsoft Fabric

An incremental loading framework processes only records that are new, changed, or
deleted since the last successful run. It avoids repeatedly loading and processing the
entire source dataset.

## Short Interview Answer

> An incremental loading framework identifies and processes only data that changed
> after the previous successful run. I design it as a metadata-driven framework that
> stores the source, target, business key, watermark column, load strategy, and last
> successful watermark for each table.
>
> At the start of a run, the framework reads the previous watermark and captures a fixed
> upper watermark from the source. It extracts records greater than the previous
> watermark and less than or equal to the upper watermark. After validation and
> deduplication, it applies the configured strategy, such as append, upsert using Delta
> `MERGE`, SCD Type 1, SCD Type 2, or CDC with delete handling.
>
> The framework must be idempotent, auditable, and restartable. It updates the watermark
> only after the target write and required validations succeed. If the run fails, the
> previous watermark remains unchanged so the same batch can be retried safely.

## Is Incremental Loading a Bronze or Silver Responsibility?

Incremental loading is applicable to **both Bronze and Silver**, but each layer has a
different responsibility.

| Layer | Incremental responsibility | Typical implementation |
|---|---|---|
| **Bronze** | Incrementally capture source data without applying business transformations. Preserve source fidelity and ingestion history. | Watermark extraction, CDC ingestion, append-only batches, source operation codes, ingestion timestamp, run ID, and source file metadata. |
| **Silver** | Incrementally process Bronze changes and apply cleansing, deduplication, business keys, deletes, and history rules. | Delta `MERGE`, upsert, SCD Type 1, SCD Type 2, deduplication, late-arriving data handling, and data-quality validation. |
| **Gold** | Optionally refresh reporting outputs incrementally from approved Silver tables. Gold should not introduce duplicated business transformation logic. | Incremental publishing, partition refresh, or aggregation refresh where supported. |

### Recommended Fabric Pattern

```text
Source systems
      |
      | Incremental extraction using watermark, CDC, or files
      v
Bronze
  - Preserve source data
  - Append ingestion batches or CDC events
  - Add ingestion metadata
      |
      | Process only new Bronze batches or changed records
      v
Silver
  - Clean and standardize
  - Deduplicate
  - Apply business keys
  - Upsert, delete, SCD Type 1, or SCD Type 2
  - Validate before committing
      |
      v
Gold
  - Publish approved reporting datasets
  - Refresh incrementally where appropriate
```

The key distinction is:

- **Bronze incremental loading** answers: "Which source records or events should be
  captured?"
- **Silver incremental loading** answers: "How should those changes affect the governed
  target table?"

SCD Type 2 normally belongs in **Silver** because it is a dimensional modeling and
history-preservation rule, not a raw-ingestion rule.

## Core Processing Flow

```text
Metadata / Control Table
          |
          v
Read previous successful watermark
          |
          v
Capture fixed upper watermark
          |
          v
Extract:
watermark > previous watermark
AND watermark <= upper watermark
          |
          v
Validate and deduplicate
          |
          v
Apply configured strategy
  - Append
  - Upsert / MERGE
  - SCD Type 1
  - SCD Type 2
  - CDC with deletes
          |
          v
Validate target and write audit data
          |
     +----+----+
     |         |
  Success    Failure
     |         |
Update       Keep previous
watermark    watermark
```

## Framework Components

| Component | Purpose |
|---|---|
| Configuration table | Stores source, target, keys, watermark, strategy, and processing options. |
| Watermark control | Stores the last successfully processed timestamp, sequence, or version. |
| Orchestration pipeline | Reads metadata, resolves dependencies, and invokes reusable processing logic. |
| Ingestion logic | Reads only the bounded source change set. |
| Transformation logic | Cleans, standardizes, joins, and applies business rules in Silver. |
| Delta write logic | Applies append, upsert, delete, SCD Type 1, or SCD Type 2 behavior. |
| Validation framework | Checks keys, duplicates, accepted values, row counts, and referential integrity. |
| Audit table | Records run status, timestamps, watermarks, row counts, and errors. |
| Recovery logic | Supports safe retries, replay, and controlled backfills. |

## Example Configuration

| Source | Target | Business key | Watermark column | Strategy | Delete handling |
|---|---|---|---|---|---|
| Sales | FactSales | OrderLineId | ModifiedDate | Upsert | Soft delete |
| Customer | DimCustomer | CustomerId | ModifiedDate | SCD Type 2 | Expire current row |
| Transactions | FactTransactions | TransactionId | SequenceId | Append | Not applicable |
| Product | DimProduct | ProductId | ModifiedDate | SCD Type 1 | Source delete flag |

Useful metadata fields include:

- Source and target workspace, lakehouse, schema, and table
- Business key columns
- Watermark column and data type
- Previous successful watermark
- Load strategy
- Tracked columns for change detection
- Delete-handling strategy
- Deduplication order column
- Data-quality rule set
- Dependency order
- Active flag

## Load Strategies

### Append

Insert all records in the incremental batch. This works for immutable events and
transactions when records are never updated.

### Upsert

Update a target row when its business key exists and insert it when it does not exist.
In Delta Lake, this is normally implemented using `MERGE`.

```sql
MERGE INTO silver.target AS target
USING staged_changes AS source
ON target.BusinessKey = source.BusinessKey
WHEN MATCHED THEN
  UPDATE SET *
WHEN NOT MATCHED THEN
  INSERT *;
```

An upsert is commonly implemented with `MERGE`, but `MERGE` can also support deletes,
conditional updates, SCD Type 1, and SCD Type 2 patterns.

### SCD Type 1

Overwrite changed attributes when history is not required.

### SCD Type 2

Expire the current dimension row and insert a new version when a tracked attribute
changes. This preserves history and is normally implemented in Silver.

### CDC

Apply source insert, update, and delete events using an operation code or change type.
CDC is usually the most reliable method for capturing hard deletes.

## Watermark Pattern

Assume:

- Previous successful watermark: `2026-09-28 00:00:00`
- Upper watermark captured at run start: `2026-09-29 00:00:00`

The extraction boundary is:

```sql
WHERE ModifiedDate >  '2026-09-28 00:00:00'
  AND ModifiedDate <= '2026-09-29 00:00:00'
```

Capturing an upper boundary prevents records arriving during the run from creating an
unbounded or inconsistent batch. The next run begins from the successfully committed
upper watermark.

For timestamp watermarks, consider ties and source timestamp precision. A monotonically
increasing sequence, source version, or CDC position is safer when available. If only a
timestamp exists, use a deterministic tie-breaker or a small overlap window followed by
deduplication.

## Delete Handling

A watermark on `ModifiedDate` does not detect a hard-deleted source row unless the source
also exposes deletion information. Common approaches are:

1. Process CDC delete events.
2. Read a source soft-delete indicator.
3. Compare source and target business keys.
4. Compare periodic source snapshots.

The selected approach should be stored in metadata because delete behavior varies by
table.

## Reliability Requirements

### Idempotency

Rerunning the same batch must not create duplicates or corrupt history. Use stable
business keys, deterministic deduplication, batch identifiers, and transactional Delta
writes.

### Watermark Commit Rule

Update the control-table watermark only after:

1. The source batch is read successfully.
2. Mandatory validation succeeds.
3. The target write commits successfully.
4. Post-write checks succeed.

If any step fails, retain the previous successful watermark.

### Auditability

Capture at least:

- Run ID
- Source and target
- Previous and upper watermarks
- Start and end timestamps
- Rows read
- Rows inserted
- Rows updated
- Rows deleted or expired
- Rows rejected
- Execution status
- Error details

### Backfill and Replay

Allow the orchestration pipeline to override the normal watermark range for a controlled
backfill. Record the override in the audit log and preserve idempotent target behavior.

## Important Design Decisions

| Question | Design consideration |
|---|---|
| Does the source expose CDC? | Prefer CDC when inserts, updates, and deletes must all be captured reliably. |
| Is the source append-only? | Use append with duplicate protection. |
| Are existing records updated? | Use an upsert or an SCD strategy. |
| Is historical analysis required? | Use SCD Type 2 in Silver. |
| Can records be hard-deleted? | Use CDC, snapshots, or explicit key comparison. |
| Can timestamps be duplicated? | Add a tie-breaker or overlap and deduplicate. |
| Can late data arrive? | Use an overlap window and deterministic replay logic. |
| Is the source available through a OneLake shortcut? | Consider virtualization instead of physically copying data into Bronze. |

## Incremental Loading vs. SCD Type 2

| Incremental loading | SCD Type 2 |
|---|---|
| Determines which records need processing. | Determines how dimension changes are stored. |
| Applies to facts, dimensions, events, and files. | Primarily applies to dimensions. |
| Can use append, upsert, CDC, SCD1, or SCD2. | Expires the current row and inserts a new historical version. |
| Applies across Bronze and Silver with different responsibilities. | Normally implemented in Silver. |

Therefore, SCD Type 2 is one target-processing strategy within the wider incremental
loading framework.

## Whiteboard Closing Statement

> In Microsoft Fabric, incremental loading applies to both Bronze and Silver. Bronze
> incrementally captures source changes with minimal modification, while Silver
> incrementally applies cleansing, deduplication, upserts, deletes, and SCD history.
> The framework should be metadata-driven, idempotent, auditable, restartable, and update
> its watermark only after a successful validated commit.

## References

- [Fabric medallion lakehouse architecture](https://learn.microsoft.com/fabric/onelake/onelake-medallion-lakehouse-architecture)
- [OneLake shortcuts](https://learn.microsoft.com/fabric/onelake/onelake-shortcuts)
- [Copy activity in Fabric](https://learn.microsoft.com/fabric/data-factory/copy-data-activity)
- [Delta Lake table maintenance](https://learn.microsoft.com/fabric/data-engineering/lakehouse-table-maintenance)
- [Slowly changing dimensions](./slowly-changing-dimensions.md)
