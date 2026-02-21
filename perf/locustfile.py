from __future__ import annotations

import os

from locust import HttpUser, LoadTestShape, between, task


API_KEY = os.getenv("DS_API_KEY", "test-key")
DASHBOARD_USER = os.getenv("DASHBOARD_USER", "admin")
DASHBOARD_PASS = os.getenv("DASHBOARD_PASS", "replace-me")

# ── Shared booking payload ────────────────────────────────────────────
_BOOKING_RECORDS = [
    {
        "lead_time": 50,
        "arrival_date_year": 2025,
        "arrival_date_week_number": 25,
        "arrival_date_day_of_month": 16,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "booking_changes": 1,
        "days_in_waiting_list": 0,
        "adr": 115.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1,
        "hotel": "Resort Hotel",
        "meal": "BB",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "deposit_type": "No Deposit",
        "customer_type": "Transient",
    }
]

_HEADERS = {"x-api-key": API_KEY}


# ─────────────────────────────────────────────────────────────────────
# User classes
# ─────────────────────────────────────────────────────────────────────


class InferenceUser(HttpUser):
    """Models a client calling inference endpoints."""

    wait_time = between(0.1, 0.5)

    @task(3)
    def health(self):
        self.client.get("/health", name="GET /health")

    @task(1)
    def ready(self):
        self.client.get("/ready", name="GET /ready")

    @task(4)
    def decide(self):
        self.client.post(
            "/decide",
            json={"records": _BOOKING_RECORDS},
            headers=_HEADERS,
            name="POST /decide",
        )

    @task(3)
    def predict_proba(self):
        self.client.post(
            "/predict_proba",
            json={"records": _BOOKING_RECORDS},
            headers=_HEADERS,
            name="POST /predict_proba",
        )

    @task(2)
    def decide_v1(self):
        self.client.post(
            "/v1/decide",
            json={"records": _BOOKING_RECORDS},
            headers=_HEADERS,
            name="POST /v1/decide",
        )

    @task(2)
    def decide_v2(self):
        self.client.post(
            "/v2/decide",
            json={"records": _BOOKING_RECORDS},
            headers=_HEADERS,
            name="POST /v2/decide",
        )

    @task(1)
    def predict_proba_v1(self):
        self.client.post(
            "/v1/predict_proba",
            json={"records": _BOOKING_RECORDS},
            headers=_HEADERS,
            name="POST /v1/predict_proba",
        )


class DashboardUser(HttpUser):
    """Models an ops user interacting with the dashboard auth flow."""

    wait_time = between(1, 3)

    def on_start(self):
        resp = self.client.post(
            "/auth/login",
            json={"username": DASHBOARD_USER, "password": DASHBOARD_PASS},
            name="POST /auth/login",
        )
        data = resp.json() if resp.status_code == 200 else {}
        self._token = data.get("token", "")

    @task(2)
    def dashboard_experiments(self):
        self.client.get(
            "/dashboard/api/experiments",
            headers={"Authorization": f"Bearer {self._token}"},
            name="GET /dashboard/api/experiments",
        )

    @task(1)
    def dashboard_metrics(self):
        self.client.get(
            "/dashboard/api/metrics",
            headers={"Authorization": f"Bearer {self._token}"},
            name="GET /dashboard/api/metrics",
        )


class ChatUser(HttpUser):
    """Models a user calling the LLM chat endpoint."""

    wait_time = between(2, 6)

    @task
    def chat(self):
        self.client.post(
            "/chat",
            json={"message": "What is the cancellation probability threshold?"},
            headers=_HEADERS,
            name="POST /chat",
        )


# ─────────────────────────────────────────────────────────────────────
# Production load shape: ramp-up → sustained → teardown
#
#   0–2 min   : ramp from 0 → 50 users   (smoke / warm-up)
#   2–7 min   : ramp from 50 → 200 users  (ramp-up)
#   7–17 min  : hold at 200 users         (sustained load)
#   17–20 min : ramp from 200 → 400 users (peak / stress)
#   20–25 min : hold at 400 users         (peak sustained)
#   25–27 min : ramp down to 0            (teardown)
# ─────────────────────────────────────────────────────────────────────
class ProductionLoadShape(LoadTestShape):
    stages = [
        {"duration": 120,  "users": 50,  "spawn_rate": 1},
        {"duration": 420,  "users": 200, "spawn_rate": 5},
        {"duration": 1020, "users": 200, "spawn_rate": 5},
        {"duration": 1200, "users": 400, "spawn_rate": 10},
        {"duration": 1500, "users": 400, "spawn_rate": 10},
        {"duration": 1620, "users": 0,   "spawn_rate": 20},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None

