"""Tests for src.guest_store — GuestStore CRUD operations."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine

from src.guest_store import GuestStore, init_guest_store, get_guest_store


def _make_engine(tmp_path: Path):
    db_path = tmp_path / "guests.db"
    return create_engine(f"sqlite:///{db_path.as_posix()}", future=True)


@pytest.fixture()
def store(tmp_path: Path) -> GuestStore:
    """Create a fresh GuestStore with an isolated SQLite DB."""
    engine = _make_engine(tmp_path)
    s = GuestStore(engine)
    s.metadata.create_all(engine)
    return s


def _sample_guest(**overrides) -> dict:
    base = {
        "first_name": "Ahmet",
        "last_name": "Yılmaz",
        "email": "ahmet@example.com",
        "phone": "+905551234567",
        "nationality": "TUR",
        "identity_no": "12345678901",
        "gender": "M",
        "vip_status": False,
        "hotel": "Resort Hotel",
        "lead_time": 45,
        "deposit_type": "No Deposit",
        "market_segment": "Online TA",
        "adults": 2,
        "children": 0,
        "babies": 0,
        "stays_in_week_nights": 3,
        "stays_in_weekend_nights": 1,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "adr": 120.50,
    }
    base.update(overrides)
    return base


class TestCreateGuest:
    """GuestStore.create_guest inserts and returns the new row."""

    def test_basic_create(self, store: GuestStore) -> None:
        result = store.create_guest(_sample_guest())
        assert result["first_name"] == "Ahmet"
        assert result["last_name"] == "Yılmaz"
        assert result["id"] is not None
        assert result["created_at"] is not None

    def test_create_with_risk_score(self, store: GuestStore) -> None:
        guest = _sample_guest(risk_score=0.72, risk_label="high")
        result = store.create_guest(guest)
        assert result["risk_score"] == 0.72
        assert result["risk_label"] == "high"


class TestListGuests:
    """GuestStore.list_guests with search/limit/offset."""

    def test_list_empty(self, store: GuestStore) -> None:
        assert store.list_guests() == []

    def test_list_returns_all(self, store: GuestStore) -> None:
        store.create_guest(_sample_guest(first_name="Ali"))
        store.create_guest(_sample_guest(first_name="Veli"))
        guests = store.list_guests()
        assert len(guests) == 2

    def test_list_respects_limit(self, store: GuestStore) -> None:
        for i in range(5):
            store.create_guest(_sample_guest(first_name=f"Guest{i}"))
        assert len(store.list_guests(limit=3)) == 3

    def test_list_search_by_name(self, store: GuestStore) -> None:
        store.create_guest(_sample_guest(first_name="Mehmet"))
        store.create_guest(_sample_guest(first_name="Zeynep"))
        results = store.list_guests(search="Mehmet")
        assert len(results) == 1
        assert results[0]["first_name"] == "Mehmet"


class TestGetGuest:
    """GuestStore.get_guest retrieves by ID."""

    def test_found(self, store: GuestStore) -> None:
        created = store.create_guest(_sample_guest())
        found = store.get_guest(created["id"])
        assert found is not None
        assert found["email"] == "ahmet@example.com"

    def test_not_found(self, store: GuestStore) -> None:
        assert store.get_guest(9999) is None


class TestCountGuests:
    """GuestStore.count_guests returns correct counts."""

    def test_count_empty(self, store: GuestStore) -> None:
        assert store.count_guests() == 0

    def test_count_with_search(self, store: GuestStore) -> None:
        store.create_guest(_sample_guest(first_name="Ali"))
        store.create_guest(_sample_guest(first_name="Veli"))
        assert store.count_guests() == 2
        assert store.count_guests(search="Ali") == 1


class TestDeleteGuest:
    """GuestStore.delete_guest removes by ID."""

    def test_delete_existing(self, store: GuestStore) -> None:
        created = store.create_guest(_sample_guest())
        assert store.delete_guest(created["id"]) is True
        assert store.get_guest(created["id"]) is None

    def test_delete_nonexistent(self, store: GuestStore) -> None:
        assert store.delete_guest(9999) is False


class TestModuleLevel:
    """Module-level init/get helpers."""

    def test_get_without_init_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.guest_store as mod

        monkeypatch.setattr(mod, "_guest_store", None)
        with pytest.raises(RuntimeError):
            get_guest_store()

    def test_init_and_get(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        init_guest_store(engine)
        s = get_guest_store()
        assert isinstance(s, GuestStore)
