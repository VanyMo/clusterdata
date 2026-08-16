#!/usr/bin/env python3
"""Compact one Alibaba pod+server hour into an OR-friendly scheduling snapshot.

The compact snapshot deliberately keeps server-level topology so later experiments
can model both ASW locality and GPU fragmentation without retaining the large raw
pod table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--hour", type=int, required=True)
    parser.add_argument("--raw", required=True, help="Directory produced by slice_hour.py")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not 0 <= args.day <= 184:
        raise SystemExit("--day must be in 0..184")
    if not 0 <= args.hour <= 23:
        raise SystemExit("--hour must be in 0..23")

    sid = f"d{args.day:03d}h{args.hour:02d}"
    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pod = raw / f"pod_d{args.day:03d}_h{args.hour:02d}_p000.parquet"
    server = raw / f"server_d{args.day:03d}_h{args.hour:02d}_p000.parquet"
    range_manifest = raw / f"manifest_{sid}.json"
    for path in (pod, server, range_manifest):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    jobs_out = out / f"jobs_{sid}.parquet"
    placements_out = out / f"placements_{sid}.parquet"
    servers_out = out / f"servers_{sid}.parquet"

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=2")
    con.execute(
        f"""
        CREATE TEMP VIEW pod AS
        SELECT *
        FROM read_parquet('{sql_path(pod)}')
        WHERE gpu_request > 0 AND server_id IS NOT NULL;

        CREATE TEMP VIEW server AS
        SELECT *
        FROM read_parquet('{sql_path(server)}');

        CREATE TEMP VIEW joined AS
        SELECT
            COALESCE(p.workload_id, 'pod:' || p.pod_id) AS job_id,
            p.workload_id,
            p.pod_id,
            p.server_id,
            COALESCE(s.cluster_id, p.cluster_id) AS cluster_id,
            s.asw_id,
            p.gpu_spec_public,
            p.priority_class,
            p.job_type_public,
            p.model_type_public,
            p.is_genai_request,
            CAST(p.gpu_request AS DOUBLE) AS gpu_request
        FROM pod p
        LEFT JOIN server s USING (server_id);
        """
    )

    # One row per job/server placement. This is the smallest table that still
    # preserves observed server-level fragmentation and ASW placement.
    con.execute(
        f"""
        COPY (
            SELECT
                {args.day}::INTEGER AS day,
                {args.hour}::INTEGER AS hour,
                job_id,
                workload_id,
                server_id,
                cluster_id,
                asw_id,
                gpu_spec_public,
                SUM(gpu_request)::DOUBLE AS allocated_gpu,
                COUNT(DISTINCT pod_id)::BIGINT AS pod_count
            FROM joined
            GROUP BY job_id, workload_id, server_id, cluster_id, asw_id, gpu_spec_public
            ORDER BY job_id, server_id
        ) TO '{sql_path(placements_out)}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # One row per server with observed occupancy. Capacity remains explicit, so
    # counterfactual schedulers can rebuild free capacity rather than trust an
    # over-aggregated ASW snapshot.
    con.execute(
        f"""
        COPY (
            WITH occ AS (
                SELECT server_id,
                       SUM(gpu_request)::DOUBLE AS observed_requested_gpu,
                       COUNT(DISTINCT job_id)::BIGINT AS observed_job_count,
                       COUNT(DISTINCT pod_id)::BIGINT AS observed_pod_count
                FROM joined
                GROUP BY server_id
            )
            SELECT
                {args.day}::INTEGER AS day,
                {args.hour}::INTEGER AS hour,
                s.server_id,
                s.cluster_id,
                s.asw_id,
                s.gpu_spec_public,
                CAST(s.gpu_count AS DOUBLE) AS gpu_capacity,
                COALESCE(o.observed_requested_gpu, 0.0)::DOUBLE AS observed_requested_gpu,
                GREATEST(CAST(s.gpu_count AS DOUBLE) - COALESCE(o.observed_requested_gpu, 0.0), 0.0)::DOUBLE AS observed_free_gpu,
                COALESCE(o.observed_job_count, 0)::BIGINT AS observed_job_count,
                COALESCE(o.observed_pod_count, 0)::BIGINT AS observed_pod_count
            FROM server s
            LEFT JOIN occ o USING (server_id)
            ORDER BY s.cluster_id, s.asw_id, s.server_id
        ) TO '{sql_path(servers_out)}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    # Job-level demand plus observed topology entropy. Entropy is computed only
    # over placement with known ASW IDs; known_asw_fraction/topology_eligible make
    # this explicit so experiments can filter without silently biasing results.
    con.execute(
        f"""
        COPY (
            WITH base AS (
                SELECT
                    job_id,
                    ANY_VALUE(workload_id) AS workload_id,
                    SUM(gpu_request)::DOUBLE AS gpu_demand,
                    COUNT(DISTINCT pod_id)::BIGINT AS pod_count,
                    COUNT(DISTINCT server_id)::BIGINT AS server_count,
                    COUNT(DISTINCT asw_id) FILTER (WHERE asw_id IS NOT NULL)::BIGINT AS asw_count,
                    SUM(gpu_request) FILTER (WHERE asw_id IS NOT NULL)::DOUBLE AS known_asw_gpu,
                    COUNT(DISTINCT gpu_spec_public)::BIGINT AS gpu_spec_count,
                    ANY_VALUE(priority_class) AS priority_class,
                    ANY_VALUE(job_type_public) AS job_type_public,
                    ANY_VALUE(model_type_public) AS model_type_public,
                    BOOL_OR(is_genai_request) AS is_genai_request
                FROM joined
                GROUP BY job_id
            ),
            asw_alloc AS (
                SELECT job_id, asw_id, SUM(gpu_request)::DOUBLE AS g
                FROM joined
                WHERE asw_id IS NOT NULL
                GROUP BY job_id, asw_id
            ),
            entropy AS (
                SELECT
                    a.job_id,
                    -SUM((a.g / b.known_asw_gpu) * LN(a.g / b.known_asw_gpu))::DOUBLE AS observed_entropy_nats
                FROM asw_alloc a
                JOIN base b USING (job_id)
                WHERE b.known_asw_gpu > 0 AND a.g > 0
                GROUP BY a.job_id
            )
            SELECT
                {args.day}::INTEGER AS day,
                {args.hour}::INTEGER AS hour,
                b.job_id,
                b.workload_id,
                b.gpu_demand,
                b.pod_count,
                b.server_count,
                b.asw_count,
                b.gpu_spec_count,
                CASE WHEN b.gpu_demand > 0 THEN COALESCE(b.known_asw_gpu, 0.0) / b.gpu_demand ELSE 0.0 END::DOUBLE AS known_asw_fraction,
                (COALESCE(b.known_asw_gpu, 0.0) >= b.gpu_demand - 1e-9) AS topology_eligible,
                COALESCE(e.observed_entropy_nats, 0.0)::DOUBLE AS observed_entropy_nats,
                b.priority_class,
                b.job_type_public,
                b.model_type_public,
                b.is_genai_request
            FROM base b
            LEFT JOIN entropy e USING (job_id)
            ORDER BY b.job_id
        ) TO '{sql_path(jobs_out)}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )

    def rows(path: Path) -> int:
        return int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{sql_path(path)}')").fetchone()[0])

    def bytes_(path: Path) -> int:
        return path.stat().st_size

    manifest = {
        "slice_id": sid,
        "day": args.day,
        "hour": args.hour,
        "raw_range_manifest": json.loads(range_manifest.read_text(encoding="utf-8")),
        "derived": {
            "jobs": {"file": jobs_out.name, "rows": rows(jobs_out), "bytes": bytes_(jobs_out)},
            "placements": {"file": placements_out.name, "rows": rows(placements_out), "bytes": bytes_(placements_out)},
            "servers": {"file": servers_out.name, "rows": rows(servers_out), "bytes": bytes_(servers_out)},
        },
    }
    manifest["derived_bytes_total"] = sum(v["bytes"] for v in manifest["derived"].values())
    manifest_path = out / f"manifest_or_{sid}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
