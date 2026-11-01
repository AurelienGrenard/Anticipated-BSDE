"""Keep paper tables, notebooks and retained runs tied to the published implementation."""

import base64
import hashlib
import json
from pathlib import Path

from initial_margin.compare_risk_estimators import aggregate as aggregate_methods
from initial_margin.run_campaign import aggregate as aggregate_campaign


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'initial_margin' / 'results'


def test_all_main_artifacts_record_the_current_solver_sources() -> None:
    """Require complete source fingerprints, not just a potentially stale Git revision."""
    expected = {str(p.relative_to(ROOT))
                for folder in ('hermite', 'initial_margin')
                for p in (ROOT / folder).rglob('*.py')}
    for name in ('atm_call_experiment.json', 'atm_butterfly_experiment.json',
                 'deterministic_reference.json', 'cvar_method_comparison.json'):
        runtime = json.loads((RESULTS / name).read_text())['runtime']
        assert runtime['python'] and runtime['torch'] and runtime['device']
        assert set(runtime['source_sha256']) == expected
        for relative, digest in runtime['source_sha256'].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_complete_paper_campaigns_use_degree_two_and_full_law_nodes() -> None:
    """Rebuild both ten-seed tables and check their common numerical configuration."""
    for product in ('call', 'butterfly'):
        payload = json.loads((RESULTS / f'atm_{product}_experiment.json').read_text())
        assert payload['complete'] is True
        assert len(payload['rows']) == 30
        assert payload['summary'] == aggregate_campaign(payload['rows'])
        assert {(r['time_steps'], r['seed']) for r in payload['rows']} == {
            (n, seed) for n in (50, 100, 200) for seed in range(42, 52)}
        for row in payload['rows']:
            assert row['config']['product'] == product
            assert row['config']['hermite_degree'] == 2
            assert row['config']['paths'] == 200_000
            assert row['config']['cvar_quadrature_nodes'] == 256
            assert row['risk_estimator']['method'] == 'quadrature'
            assert row['risk_estimator']['quadrature_nodes'] == 256


def test_method_comparison_uses_the_same_quadrature_run() -> None:
    """Check the independent audit reproduces the main campaign on each common seed."""
    comparison = json.loads((RESULTS / 'cvar_method_comparison.json').read_text())
    assert len(comparison['rows']) == 6
    assert comparison['summary'] == aggregate_methods(comparison['rows'])
    for row in comparison['rows']:
        campaign = json.loads((RESULTS / f"atm_{row['product']}_experiment.json").read_text())
        same = next(r for r in campaign['rows']
                    if r['seed'] == row['seed'] and r['time_steps'] == row['time_steps'])
        quadrature = row['methods']['quadrature']
        assert quadrature['settings']['quadrature_nodes'] == 256
        assert abs(quadrature['price'] - same['metrics']['direct_mkabsde']['price']) < 1e-12
        assert all(m['gpu_seconds'] > 0 for m in row['methods'].values())


def test_peng_yang_notebook_embeds_the_current_six_figures() -> None:
    """Detect stale presentation outputs without rerunning any Peng--Yang calculation."""
    folder = ROOT / 'peng_yang'
    figures = {hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (folder / 'figures').glob('*.png')}
    notebook = json.loads((folder / 'notebooks' / 'peng_yang_results.ipynb').read_text())
    embedded = []
    for cell in notebook['cells']:
        for output in cell.get('outputs', []):
            png = output.get('data', {}).get('image/png')
            if png:
                embedded.append(hashlib.sha256(base64.b64decode(''.join(png))).hexdigest())
    assert len(embedded) == 6
    assert set(embedded) == figures


def test_base_horizon_sweep_matches_the_main_campaign() -> None:
    """Require the independent horizon sweep to recover all three base-horizon prices."""
    rows = [json.loads(line) for line in
            (RESULTS / 'margin_periods' / 'runs.jsonl').read_text().splitlines()]
    for row in rows:
        config = row['config']
        assert config['cvar_quadrature_nodes'] == 256
        if config['margin_period'] != .02:
            continue
        campaign = json.loads((RESULTS / f"atm_{config['product']}_experiment.json").read_text())
        same = next(r for r in campaign['rows']
                    if r['time_steps'] == config['time_steps'] and r['seed'] == config['seed'])
        for name, main_name in [('direct', 'direct_mkabsde'),
                                ('nonlinear', 'nonlinear_proxy'), ('linear', 'linear_proxy')]:
            assert abs(row['result']['methods'][name]['price']
                       - same['metrics'][main_name]['price']) < 1e-12
