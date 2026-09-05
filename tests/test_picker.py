import random

from memescalp.models import TokenSnapshot
from memescalp.picker import build_prompt, parse_pick, random_pick


def snaps():
    return [
        TokenSnapshot(mint="MINT_A", symbol="AAA", price_usd=0.001,
                      liquidity_usd=50_000, volume_h24=1e6,
                      price_change_h1=2.0, price_change_h24=-5.0),
        TokenSnapshot(mint="MINT_B", symbol="BBB", price_usd=0.5,
                      liquidity_usd=200_000, volume_h24=3e6,
                      price_change_h1=-1.0, price_change_h24=10.0),
    ]


def test_prompt_contains_every_candidate_and_its_data():
    prompt = build_prompt(snaps(), 1_700_000_000.0)
    assert "MINT_A" in prompt and "MINT_B" in prompt
    assert "$50,000" in prompt
    assert "+2.00%" in prompt


def test_parse_clean_json():
    r = '{"mint": "MINT_B", "symbol": "BBB", "reasoning": "momentum"}'
    assert parse_pick(r, snaps()).mint == "MINT_B"


def test_parse_json_wrapped_in_prose():
    r = 'Here is my pick:\n```json\n{"mint": "MINT_A", "symbol": "AAA", "reasoning": "x"}\n```\nGood luck.'
    assert parse_pick(r, snaps()).mint == "MINT_A"


def test_parse_falls_back_to_quoted_mint():
    assert parse_pick("I would go with MINT_B here.", snaps()).mint == "MINT_B"


def test_parse_rejects_unknown_token():
    assert parse_pick('{"mint": "NOT_IN_LIST"}', snaps()) is None
    assert parse_pick("no idea", snaps()) is None


def test_random_pick_uses_candidate_list_and_logs():
    d = random_pick(snaps(), 0.0, 1800.0, rng=random.Random(42))
    assert d.arm == "random"
    assert d.mint in {"MINT_A", "MINT_B"}
    assert "random.choice" in d.prompt
