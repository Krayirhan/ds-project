"""initial_dashboard_schema

Revision ID: 221e77090362
Revises: 
Create Date: 2026-02-21 15:00:49.558379

Creates the two core DashboardStore tables:
  - experiment_runs    : one row per ML training run (champion summary)
  - model_metrics      : per-model evaluation metrics for each run

Subsequent schema changes should be added as new revisions via:
  alembic revision --autogenerate -m "describe the change"
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '221e77090362'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial dashboard schema."""
    op.create_table(
        "experiment_runs",
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column("selected_model", sa.String(256), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("expected_net_profit", sa.Float(), nullable=True),
        sa.Column("max_action_rate", sa.Float(), nullable=True),
        sa.Column("source_path", sa.String(1024), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "model_metrics",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False, index=True),
        sa.Column("model_name", sa.String(256), nullable=False),
        sa.Column("train_cv_roc_auc_mean", sa.Float(), nullable=True),
        sa.Column("train_cv_roc_auc_std", sa.Float(), nullable=True),
        sa.Column("test_roc_auc", sa.Float(), nullable=True),
        sa.Column("test_f1", sa.Float(), nullable=True),
        sa.Column("test_precision", sa.Float(), nullable=True),
        sa.Column("test_recall", sa.Float(), nullable=True),
        sa.Column("test_threshold", sa.Float(), nullable=True),
        sa.Column("n_test", sa.Integer(), nullable=True),
        sa.Column("positive_rate_test", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["experiment_runs.run_id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    """Drop dashboard schema tables."""
    op.drop_table("model_metrics")
    op.drop_table("experiment_runs")

