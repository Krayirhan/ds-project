from __future__ import annotations

import os
from locust import HttpUser, between, task


API_KEY = os.getenv("DS_API_KEY", "test-key")


class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(2)
    def health(self):
        self.client.get("/health", headers={"x-api-key": API_KEY}, name="GET /health")

    @task(1)
    def decide(self):
        payload = {
            "records": [
                {
                    "lead_time": 50,
                    "arrival_date_year": 2017,
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
        }
        self.client.post(
            "/decide", json=payload, headers={"x-api-key": API_KEY}, name="POST /decide"
        )
