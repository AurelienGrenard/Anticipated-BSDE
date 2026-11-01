"""Keep manuscript table formatting tied to the committed numerical summaries."""

import json
from pathlib import Path

from tools.export_initial_margin_tables import estimate, load_tables, scientific


def test_scientific_and_mean_formatting() -> None:
    """Check aligned decimals and the compact LaTeX uncertainty notation."""
    assert scientific(.00123) == r'1.23\times10^{-3}'
    assert estimate(.123, .00123) == r'$0.123000\pm1.23\times10^{-3}$'


def test_exports_match_retained_values() -> None:
    """Check all four tables are generated and finest-grid numbers match their JSON."""
    results = Path(__file__).resolve().parents[1]/'initial_margin'/'results'
    tables, checks = load_tables(results)
    assert set(tables) == {'call', 'butterfly', 'methods', 'margins'}
    assert tables['margins'].count(r'\\') == 6
    assert tables['methods'].count(r'\\') == 3
    for product in ('call', 'butterfly'):
        payload = json.loads((results/f'atm_{product}_experiment.json').read_text())
        value = payload['summary'][-1]['direct_mkabsde_mva_percent_of_black_scholes_mean']
        assert checks['products'][product]['direct_mva_percent'] == value
        assert f'{value:.6f}' in tables[product]
