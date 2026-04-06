# PRANA-G AI Sovereign Scientific Data Lake

Production-ready foundation for a sovereign biological data lake that downloads, cleans, labels, stores, indexes, and serves gene-sequence datasets for AI training and simulation workflows.

## What Is Implemented

- NCBI Entrez FASTA acquisition with batching, retries, exponential backoff, rate limiting, and structured logging
- Local reference-genome extraction from genome FASTA + GFF annotation
- Hybrid acquisition mode: local genome first, NCBI fallback for unmatched genes
- TXT, CSV, and TSV gene-list ingestion
- Preprocessing pipeline for deduplication, sequence normalization, invalid-character filtering, and length validation
- Rule-based stress-labeling engine with provenance and confidence metadata
- Metadata persistence in a local SQLite operational store, plus PostgreSQL schema design for production rollout
- Vector search index using k-mer embeddings with cosine similarity
- Dataset versioning and ingestion job lineage
- FastAPI service for querying sequences, datasets, vector search, health, metrics, and ingesting new gene lists
- Cron-friendly orchestration CLI for scheduled ingestion
- Unit and API tests

## Architecture

```text
Gene Lists (TXT/CSV/TSV)
          |
          v
+--------------------------+      +----------------------+
| Acquisition Layer        |<---->| NCBI Entrez API      |
| NCBI + genome extractor  |      | FASTA / GenBank      |
| async + local reference  |      +----------------------+
+------------+-------------+
             |
             +----------------------+
             |                      |
             v                      v
   +--------------------+   +----------------------+
   | Reference Genome   |   | Annotation GFF       |
   | local FASTA assets |   | gene coordinates     |
   +--------------------+   +----------------------+
             |
             v
+--------------------------+
| Raw Data Zone            |
| versioned FASTA outputs  |
| manifests, audit trail   |
+------------+-------------+
             |
             v
+--------------------------+
| Preprocessing Layer      |
| dedupe, QC, normalize    |
+------------+-------------+
             |
             v
+--------------------------+
| Labeling Engine          |
| drought/stress taxonomy  |
| provenance + confidence  |
+------+-------------+-----+
       |             |
       v             v
+-------------+   +------------------+
| Metadata DB |   | Vector Index     |
| jobs, seqs, |   | k-mer embeddings |
| labels, ver |   | cosine search    |
+------+------+   +--------+---------+
       |                   |
       +---------+---------+
                 |
                 v
        +-------------------+
        | FastAPI Service   |
        | query + ingest    |
        | metrics + health  |
        +-------------------+
```

Cross-cutting concerns:

- Dataset versioning
- Cacheable API query layer
- Failure alerts
- Health metrics
- Cron-based orchestration
- PostgreSQL production schema

## Why This Architecture

1. Raw, processed, and labeled zones are separated so every dataset version is reproducible.
2. Acquisition is async because NCBI is I/O-bound and enforces request ceilings.
3. Preprocessing and labeling are modular so future ML-based inference can be added without rewriting ingestion.
4. Metadata belongs in a relational store because lineage, filtering, and auditability are critical.
5. Similarity search belongs in a vector index rather than transactional tables.

## Storage Design

### Local operational store

The shipped runtime uses a local SQLite metadata store for immediate usability:

- `ingestion_jobs`
- `dataset_versions`
- `source_records`
- `sequence_records`
- `sequence_labels`
- `embeddings`

### PostgreSQL production schema

The production PostgreSQL schema is defined in [sql/postgres_schema.sql](sql/postgres_schema.sql).

This is the recommended production posture:

- PostgreSQL for metadata, lineage, labels, and job tracking
- object storage or mounted volumes for raw and processed sequence assets
- vector index persisted separately and refreshed per dataset version

## Repository Layout

```text
.
├── config/
│   └── settings.yaml
├── data/
│   ├── intake/
│   ├── labeled/
│   ├── metadata/
│   ├── processed/
│   ├── raw/
│   └── vector/
├── sql/
│   └── postgres_schema.sql
├── src/prana_ai_data_lake/
│   ├── acquisition/
│   ├── api/
│   ├── cli/
│   ├── common/
│   ├── labeling/
│   ├── monitoring/
│   ├── orchestration/
│   ├── preprocessing/
│   ├── storage/
│   └── config.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Configuration

Main config file: [config/settings.yaml](config/settings.yaml)

Important sections:

- `ncbi`: API behavior, rate limiting, concurrency, organism filters
- `reference_genome`: local genome FASTA, GFF annotation, and output paths
- `preprocessing`: min/max length and allowed characters
- `labeling`: taxonomy rules
- `storage`: metadata DB path, PostgreSQL schema path, vector index path
- `orchestration`: cron settings and intake directory
- `api`: bind host, port, cache TTL
- `monitoring`: webhook alert configuration

Optional environment variables:

```powershell
$env:PRANA_NCBI_API_KEY="your_ncbi_api_key"
$env:PRANA_NCBI_EMAIL="you@example.com"
$env:PRANA_LOG_LEVEL="INFO"
```

## Recommended Mode

For PRANA-G AI and your data-engineering role, the recommended operational order is:

1. `genome` mode for species where you have a trusted reference FASTA + GFF
2. `hybrid` mode when you want local speed plus NCBI fallback for unmatched genes
3. `ncbi` mode when no local reference assets are available

Why:

- genome extraction is faster and more reproducible
- hybrid gives broader coverage without depending fully on API throttling
- NCBI-only is useful, but best as a fallback or enrichment path

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## End-to-End Pipeline Run

This runs acquisition -> preprocessing -> labeling -> metadata persistence -> vector indexing.

```powershell
python -m prana_ai_data_lake.cli.run_pipeline `
  --input "assests\drought_genes.txt" `
  --config "config/settings.yaml" `
  --limit-genes 10 `
  --max-records-per-gene 3 `
  --notes "drought pilot dataset"
```

## Genome Mode Run

Use this when you have a local genome FASTA and annotation GFF.

```powershell
python -m prana_ai_data_lake.cli.run_pipeline `
  --input "assests\drought_genes.txt" `
  --config "config/settings.yaml" `
  --mode genome `
  --reference-fasta-path "C:\path\to\genome.fna" `
  --reference-gff-path "C:\path\to\annotation.gff"
```

## Hybrid Mode Run

This is the best practical mode for your project when local reference assets are available.

```powershell
python -m prana_ai_data_lake.cli.run_pipeline `
  --input "assests\drought_genes.txt" `
  --config "config/settings.yaml" `
  --mode hybrid `
  --reference-fasta-path "C:\path\to\genome.fna" `
  --reference-gff-path "C:\path\to\annotation.gff"
```

Outputs:

- raw FASTA under `data/raw/ncbi/fasta/<run_id>/`
- genome-local FASTA under `data/raw/genome_local/fasta/<run_id>/`
- run manifests under `data/raw/ncbi/manifests/`
- processed JSONL under `data/processed/sequences/`
- labeled JSONL under `data/labeled/datasets/`
- metadata DB under `data/metadata/`
- vector index under `data/vector/`

## Acquisition-Only Run

```powershell
python -m prana_ai_data_lake.cli.download_ncbi_fasta `
  --input "assests\drought_genes.txt" `
  --config "config/settings.yaml" `
  --limit-genes 5 `
  --max-records-per-gene 3
```

## Serve The API

```powershell
python -m prana_ai_data_lake.cli.serve_api --config "config/settings.yaml"
```

Default base URL:

```text
http://127.0.0.1:8000
```

## API Endpoints

- `GET /health`
- `GET /metrics`
- `GET /sequences?gene_symbol=DREB1A`
- `GET /sequences?label=drought_tolerant`
- `GET /datasets`
- `GET /datasets/{version_id}`
- `POST /ingest/gene-lists`
- `POST /search/vector`

### Example ingest request

```json
{
  "gene_symbols": ["DREB1A", "DREB1B", "NAC1"],
  "organism": "Viridiplantae",
  "max_records_per_gene": 2,
  "notes": "api submitted batch",
  "mode": "hybrid",
  "reference_fasta_path": "C:/path/to/genome.fna",
  "reference_gff_path": "C:/path/to/annotation.gff"
}
```

### Example vector search request

```json
{
  "sequence": "ATGCATGCATGCATGCATGC",
  "dataset_version": "dataset_20260318T072506Z",
  "top_k": 5
}
```

## Quick Ingest Runner (for cron/Task Scheduler)

The repo includes a thin runner for automation: `scripts/run_ingest.py`.

```powershell
python scripts/run_ingest.py `
  --genes "assests/drought_genes.txt" `
  --config "config/settings.yaml" `
  --mode ncbi `
  --notes "nightly ncbi ingest"
```

Windows Task Scheduler example (daily 2 AM):

```powershell
schtasks /Create /SC DAILY /TN "prana_ingest" /TR "cmd /c set PYTHONPATH=%CD%\\src&& python scripts\\run_ingest.py --genes assests\\drought_genes.txt --config config\\settings.yaml --mode ncbi --notes nightly" /ST 02:00
```

Linux cron example:

```
0 2 * * * cd /path/to/Database && PYTHONPATH=$(pwd)/src python scripts/run_ingest.py --genes assests/drought_genes.txt --config config/settings.yaml --mode ncbi --notes nightly >> logs/cron_ingest.log 2>&1
```

## Postgres (optional)

1) Set `storage.engine: postgres` in `config/settings.yaml`.
2) Provide a DSN in `storage.postgres_dsn` (e.g. `postgresql://user:pass@host:5432/prana_g_ai`).
3) Run any ingest; schema/migrations are applied automatically.

## Materials Project Ingestion (Data Lake expansion)

Config:

- `materials_project.base_url`: `https://api.materialsproject.org`
- `materials_project.api_key`: or set env `PRANA_MP_API_KEY`
- Output dirs: `data/raw/materials_project/json`, manifests in `data/raw/materials_project/manifests`

Run:

```powershell
python scripts/run_materials_ingest.py `
  --formulas "assests/material_formulas.txt" `
  --config "config/settings.yaml" `
  --limit-formulas 50
```

Input: TXT with one formula per line (e.g., `LiFePO4`). Output: JSON per formula + manifest/summary.

## Labeling Strategy

The current proprietary rule engine uses gene symbols and metadata keywords to assign labels such as:

- `drought_tolerant`
- `heat_resistant`
- `stress_resilient`

Each emitted label includes:

- label source
- confidence score
- matched-keyword provenance

This makes it easy to add a future ML inference layer beside the current rules.

## Caching, Versioning, and Metrics

- API query results are cached with a TTL cache
- every pipeline run creates a new dataset version
- `/metrics` exposes Prometheus-style gauges for sequences, labels, datasets, and jobs
- failed jobs emit alert events to logs and optionally to a webhook

## Validation

Run the test suite:

```powershell
python -m pytest -q
```

Current validation coverage includes:

- gene-list parsing
- FASTA parsing
- preprocessing and deduplication
- rule-based labeling
- metadata persistence
- FastAPI endpoints
- vector search

## Delivered Files Of Interest

- [src/prana_ai_data_lake/acquisition/downloader.py](src/prana_ai_data_lake/acquisition/downloader.py)
- [src/prana_ai_data_lake/acquisition/genome_local.py](src/prana_ai_data_lake/acquisition/genome_local.py)
- [src/prana_ai_data_lake/preprocessing/pipeline.py](src/prana_ai_data_lake/preprocessing/pipeline.py)
- [src/prana_ai_data_lake/labeling/engine.py](src/prana_ai_data_lake/labeling/engine.py)
- [src/prana_ai_data_lake/storage/repository.py](src/prana_ai_data_lake/storage/repository.py)
- [src/prana_ai_data_lake/storage/vector_index.py](src/prana_ai_data_lake/storage/vector_index.py)
- [src/prana_ai_data_lake/orchestration/pipeline_runner.py](src/prana_ai_data_lake/orchestration/pipeline_runner.py)
- [src/prana_ai_data_lake/api/app.py](src/prana_ai_data_lake/api/app.py)
- [sql/postgres_schema.sql](sql/postgres_schema.sql)
