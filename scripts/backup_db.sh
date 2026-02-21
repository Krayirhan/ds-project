#!/usr/bin/env bash
# backup_db.sh — PostgreSQL point-in-time backup via pg_dump
#
# Usage:
#   DATABASE_URL=postgresql://user:pass@host:5432/db \
#   BACKUP_DIR=/backups \
#   BACKUP_RETENTION_DAYS=14 \
#   ./scripts/backup_db.sh
#
# In Kubernetes, this is run as a CronJob (see deploy/k8s/backup-cronjob.yaml).
# Backups are gzip-compressed SQL dumps uploaded to the BACKUP_DIR path,
# which should be a mounted PersistentVolume or an s3-fuse mount.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────
: "${DATABASE_URL:?DATABASE_URL env var is required}"
: "${BACKUP_DIR:=/backups}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-14}"

# ── Parse connection parts from DATABASE_URL ──────────────────────────
# Supports:  postgresql+psycopg://user:pass@host:5432/dbname
#            postgresql://user:pass@host:5432/dbname
_url="${DATABASE_URL#postgresql+psycopg://}"
_url="${_url#postgresql://}"

PG_USER="${_url%%:*}"
_rest="${_url#*:}"
PG_PASS="${_rest%%@*}"
_rest="${_rest#*@}"
PG_HOST="${_rest%%:*}"
_rest="${_rest#*:}"
PG_PORT="${_rest%%/*}"
PG_DB="${_rest#*/}"
# Strip query string if present
PG_DB="${PG_DB%%\?*}"

# ── Backup ────────────────────────────────────────────────────────────
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
FILENAME="${PG_DB}_${TIMESTAMP}.sql.gz"
DEST="${BACKUP_DIR}/${FILENAME}"

mkdir -p "${BACKUP_DIR}"

echo "[backup_db] Starting backup: ${PG_DB} → ${DEST}"

PGPASSWORD="${PG_PASS}" pg_dump \
  --host="${PG_HOST}" \
  --port="${PG_PORT}" \
  --username="${PG_USER}" \
  --dbname="${PG_DB}" \
  --format=plain \
  --no-owner \
  --no-acl \
  | gzip -9 > "${DEST}"

SIZE="$(du -sh "${DEST}" | cut -f1)"
echo "[backup_db] Backup complete: ${DEST} (${SIZE})"

# ── Prune old backups ─────────────────────────────────────────────────
echo "[backup_db] Pruning backups older than ${RETENTION_DAYS} days…"
find "${BACKUP_DIR}" -name "${PG_DB}_*.sql.gz" -mtime "+${RETENTION_DAYS}" -delete
REMAINING="$(find "${BACKUP_DIR}" -name "${PG_DB}_*.sql.gz" | wc -l)"
echo "[backup_db] Retention complete. ${REMAINING} backup(s) kept."
