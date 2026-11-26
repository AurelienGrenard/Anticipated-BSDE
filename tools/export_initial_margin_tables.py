"""Export the five MVA manuscript table bodies from the retained numerical artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from initial_margin.evaluation import ARTICLE_REFERENCE_VALUES
from tools.mva_robustness import ESTIMATORS, PRODUCTS, method_comparison, spatial_comparison


def scientific(value: float) -> str:
    """Format a sample standard deviation as a compact LaTeX scientific number."""
    mantissa, exponent = f"{value:.2e}".split('e')
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def estimate(mean: float, sd: float) -> str:
    """Use six equal decimal places for all displayed MVA means."""
    return f"${mean:.6f}" + r"\pm" + scientific(sd) + "$"


def load_tables(results: Path, audit_file: Path | None = None) -> tuple[dict[str, str], dict]:
    """Build table rows and comparison quantities directly from stored summaries."""
    tables, checks = {}, {'products': {}, 'margins': []}
    reference = json.loads((results/'deterministic_reference.json').read_text())
    references = {(r['product'], r['time_steps']): r for r in reference['rows']}
    for product in ('call', 'butterfly'):
        payload = json.loads((results/f'atm_{product}_experiment.json').read_text())
        rows = []
        for row in payload['summary']:
            cells = [str(row['time_steps'])]
            for name in ('direct_mkabsde', 'nonlinear_proxy', 'linear_proxy'):
                prefix = name+'_mva_percent_of_black_scholes'
                cells.append(estimate(row[prefix+'_mean'], row[prefix+'_std']))
            rows.append(' & '.join(cells) + r'\\')
        base = payload['rows'][0]['metrics']['black_scholes_no_im']['price']
        published = ARTICLE_REFERENCE_VALUES[product]
        article = [100*(published[name+'_price']-base)/base
                   for name in ('nonlinear_proxy', 'linear_proxy')]
        rows.append(r'\hline')
        # The source prices have only four decimal places.
        rows.append(' & '.join(['Article', '--', *(rf'$\approx {v:.2f}$' for v in article)]) + r'\\')
        tables[product] = '\n'.join(rows)+'\n'
        finest = payload['summary'][-1]
        direct = finest['direct_mkabsde_mva_percent_of_black_scholes_mean']
        checks['products'][product] = {
            'direct_mva_percent': direct,
            'proxy_mva_percent': {name: finest[name+'_mva_percent_of_black_scholes_mean']
                                 for name in ('nonlinear_proxy', 'linear_proxy')},
            'relative_mva_gap_percent': {
                name: 100*(direct/finest[name+'_mva_percent_of_black_scholes_mean']-1)
                for name in ('nonlinear_proxy', 'linear_proxy')},
            'abs_price_gap_from_deterministic': abs(finest['direct_mkabsde_price_mean']
                - references[(product, finest['time_steps'])]['price']),
        }
    checks['methods'] = method_comparison(results, audit_file)
    rows = []
    for row in checks['methods']:
        cells = [row['product'].capitalize(), f"${row['margin_period']:.2f}$"]
        for method in ESTIMATORS:
            metrics = row['methods'][method]
            # The estimate and its sample SD are both in MVA percentage points.
            cells.append(f"${metrics['mean']:.4f}" + r"\pm" + f"{metrics['sd']:.4f}$")
        rows.append(' & '.join(cells) + r'\\')
    tables['methods'] = '\n'.join(rows) + '\n'
    checks['spatial'] = spatial_comparison(results)
    rows = []
    for row in checks['spatial']:
        cells = [f"${row['margin_period']:.2f}$", str(row['time_steps'])]
        for product in PRODUCTS:
            cells.extend(f"${row['products'][product][name]:.6f}$" for name in ('hermite', 'fine_grid'))
        rows.append(' & '.join(cells) + r'\\')
    tables['spatial'] = '\n'.join(rows) + '\n'
    margins = json.loads((results/'margin_periods'/'summary.json').read_text())
    rows, previous = [], None
    for row in margins:
        product = row['product']
        cells = [product.capitalize() if product != previous else '', f"${row['margin_period']:.2f}$"]
        cells.extend(f"${row['methods'][name]['mva_percent_mean']:.4f}$"
                     for name in ('direct', 'nonlinear', 'linear'))
        rows.append(' & '.join(cells)+r'\\')
        previous = product
        checks['margins'].append({'product': product, 'delta': row['margin_period'],
            'nonlinear_relative_mva_gap_percent': 100*(
                row['methods']['direct']['mva_mean']/row['methods']['nonlinear']['mva_mean']-1)})
    tables['margins'] = '\n'.join(rows)+'\n'
    return tables, checks


def main() -> None:
    """Write portable table fragments without changing the manuscript or source data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'initial_margin'/'results')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--es-audit', type=Path,
                        help='Use a fresh, complete three-horizon ES comparison instead of the retained runs.')
    args = parser.parse_args()
    tables, checks = load_tables(args.results, args.es_audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        (args.output_dir/f'{name}_rows.tex').write_text(rows)
    (args.output_dir/'numerical_checks.json').write_text(json.dumps(checks, indent=2, allow_nan=False)+'\n')
    print(f'Exported five table bodies and numerical checks to {args.output_dir}')


if __name__ == '__main__':
    main()
