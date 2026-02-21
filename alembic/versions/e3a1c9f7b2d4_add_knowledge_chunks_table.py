"""add_knowledge_chunks_table

Revision ID: e3a1c9f7b2d4
Revises: 854e7dedec10
Create Date: 2026-02-21

Adds the `knowledge_chunks` table for pgvector-backed RAG knowledge base.
Enables the pgvector extension and creates an HNSW index for sub-millisecond
cosine similarity search across policy/knowledge embeddings.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e3a1c9f7b2d4"
down_revision: Union[str, Sequence[str], None] = "854e7dedec10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBED_DIM = 768  # nomic-embed-text output dimension


def upgrade() -> None:
    # ── 1. Enable pgvector extension ─────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── 2. Create knowledge_chunks table ─────────────────────────────────────
    op.create_table(
        "knowledge_chunks",
        sa.Column("id",         sa.Integer(),      primary_key=True, autoincrement=True),
        sa.Column("chunk_id",   sa.String(50),     nullable=False),
        sa.Column("category",   sa.String(50),     nullable=False, server_default="general"),
        sa.Column("tags",       sa.JSON(),          nullable=False, server_default="[]"),
        sa.Column("title",      sa.String(200),    nullable=False),
        sa.Column("content",    sa.Text(),          nullable=False),
        sa.Column("priority",   sa.Integer(),      nullable=False, server_default="5"),
        sa.Column("is_active",  sa.Boolean(),      nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.UniqueConstraint("chunk_id", name="uq_knowledge_chunks_chunk_id"),
    )

    # ── 3. Add vector column (pgvector type — cannot use SA natively) ─────────
    op.execute(f"ALTER TABLE knowledge_chunks ADD COLUMN embedding vector({EMBED_DIM})")

    # ── 4. HNSW index for fast cosine similarity search ───────────────────────
    # m=16, ef_construction=64 are good defaults for small-medium corpora.
    # HNSW trades index build time for extremely fast query time (sub-ms).
    op.execute(
        "CREATE INDEX ix_knowledge_embedding_hnsw "
        "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.drop_table("knowledge_chunks")
    # Note: we intentionally do NOT drop the vector extension
    # as it may be used by other tables in the future.
