from __future__ import annotations

import src.chat.pipeline.context_builder as cb
import src.chat.pipeline.response_validator as rv


def test_build_customer_context_risk_levels_and_summary_paths():
    high = cb.build_customer_context(
        customer_data={
            "hotel": "City Hotel",
            "lead_time": 220,
            "deposit_type": "No Deposit",
            "previous_cancellations": 2,
            "is_repeated_guest": 1,
            "market_segment": "Online TA",
            "adults": 2,
            "children": 1,
            "stays_in_week_nights": 2,
            "stays_in_weekend_nights": 1,
        },
        risk_score=0.82,
        risk_label="high",
        retrieved_chunks_text="chunk",
    )
    assert high.risk_level_tr
    assert high.risk_percent == 82.0
    assert len(high.key_risk_factors) >= 3
    assert "City Hotel" in high.profile_summary_tr

    medium = cb.build_customer_context(
        customer_data={"lead_time": 30},
        risk_score=0.5,
        risk_label="mid",
        retrieved_chunks_text="",
    )
    assert medium.risk_percent == 50.0
    assert medium.risk_level_tr

    low = cb.build_customer_context(
        customer_data={"lead_time": 3, "deposit_type": "Non Refund"},
        risk_score=0.2,
        risk_label="low",
        retrieved_chunks_text="",
    )
    assert low.risk_percent == 20.0
    assert low.risk_level_tr
    assert len(low.key_risk_factors) >= 1


def test_validate_response_and_fallback_variants():
    good = rv.validate_response("Bu yanit Turkce ve yeterince uzundur.")
    assert good.cleaned_response
    assert good.is_valid is True

    english = rv.validate_response("The customer should cancel this booking please.")
    assert english.is_valid is False
    assert "mostly_english" in english.issues

    empty = rv.validate_response("   ")
    assert empty.is_valid is False
    assert "empty" in empty.issues
    assert "too_short" in empty.issues

    short = rv.validate_response("kisa yanit")
    assert "too_short" in short.issues

    hi = rv.fallback_response(70)
    mid = rv.fallback_response(40)
    low = rv.fallback_response(20)
    assert "%" in hi and "%" in mid and "%" in low
