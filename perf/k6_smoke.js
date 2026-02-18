import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: 20,
  duration: "2m",
  thresholds: {
    http_req_duration: ["p(95)<300", "p(99)<800"],
    http_req_failed: ["rate<0.01"],
  },
};

const BASE = __ENV.BASE_URL || "http://127.0.0.1:8000";
const API_KEY = __ENV.DS_API_KEY || "test-key";

export default function () {
  const healthRes = http.get(`${BASE}/health`, {
    headers: { "x-api-key": API_KEY },
  });
  check(healthRes, { "health status 200": (r) => r.status === 200 });

  const payload = JSON.stringify({
    records: [
      {
        lead_time: 50,
        arrival_date_year: 2017,
        arrival_date_week_number: 25,
        arrival_date_day_of_month: 16,
        stays_in_weekend_nights: 1,
        stays_in_week_nights: 2,
        adults: 2,
        children: 0,
        babies: 0,
        is_repeated_guest: 0,
        previous_cancellations: 0,
        previous_bookings_not_canceled: 0,
        booking_changes: 1,
        days_in_waiting_list: 0,
        adr: 115.0,
        required_car_parking_spaces: 0,
        total_of_special_requests: 1,
        hotel: "Resort Hotel",
        meal: "BB",
        market_segment: "Online TA",
        distribution_channel: "TA/TO",
        reserved_room_type: "A",
        assigned_room_type: "A",
        deposit_type: "No Deposit",
        customer_type: "Transient",
      },
    ],
  });

  const decideRes = http.post(`${BASE}/decide`, payload, {
    headers: { "Content-Type": "application/json", "x-api-key": API_KEY },
  });
  check(decideRes, { "decide status 200": (r) => r.status === 200 });
  sleep(0.2);
}
