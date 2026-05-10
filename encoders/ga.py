"""
GA-based FIC Encoder.

Each chromosome represents one candidate mapping for a range block:
    [domain_idx, isometry]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fic_core as core


DEFAULT_CONFIG = {
    'pop_size': 40,
    'generation': 30,
    'run': 1,
    'tournament_size': 3,
    'crossover_rate': 0.8,
    'mutation_rate': 0.15,
    'elitism': 2,
    'domain_mutation_radius': 8,
    'seed': 42,
}


def _parse_scalar(value):
    value = value.strip()
    lowered = value.lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    if lowered in ('null', 'none', '~'):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('\'"')


def load_ga_config(config_path=None):
    """
    Load a simple flat YAML config without requiring PyYAML.

    Supported format:
        key: value
    """
    if config_path is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(root_dir, 'configs', 'ga.yaml')

    config = DEFAULT_CONFIG.copy()
    if not os.path.exists(config_path):
        return config

    with open(config_path, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            if key:
                config[key] = _parse_scalar(value)
    return config


def _evaluate_gene(r_block, all_iso, gene):
    d_idx = int(gene[0])
    iso = int(gene[1])
    s, o, mse = core.evaluate_candidate(r_block, all_iso, d_idx, iso)
    if abs(s) >= 1.0:
        mse = np.inf
    return {
        'mse': float(mse),
        'd_idx': d_idx,
        'iso': iso,
        's': float(s),
        'o': float(o),
    }


def _initial_population(rng, pop_size, n_domain):
    population = np.empty((pop_size, 2), dtype=np.int64)
    population[:, 0] = rng.integers(0, n_domain, size=pop_size)
    population[:, 1] = rng.integers(0, 8, size=pop_size)
    return population


def _tournament_select(rng, population, fitness, tournament_size):
    size = min(tournament_size, len(population))
    idx = rng.choice(len(population), size=size, replace=False)
    winner = idx[int(np.argmin(fitness[idx]))]
    return population[winner].copy()


def _crossover(rng, parent_a, parent_b, crossover_rate):
    if rng.random() >= crossover_rate:
        return parent_a.copy(), parent_b.copy()
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    if rng.random() < 0.5:
        child_a[0], child_b[0] = child_b[0], child_a[0]
    else:
        child_a[1], child_b[1] = child_b[1], child_a[1]
    return child_a, child_b


def _mutate(rng, gene, n_domain, mutation_rate, domain_mutation_radius):
    if rng.random() < mutation_rate:
        step = rng.integers(-domain_mutation_radius, domain_mutation_radius + 1)
        gene[0] = int(np.clip(gene[0] + step, 0, n_domain - 1))
    if rng.random() < mutation_rate:
        gene[1] = rng.integers(0, 8)
    return gene


def _fallback_valid_candidate(r_block, all_iso, n_domain):
    best = {'mse': np.inf, 'd_idx': 0, 'iso': 0, 's': 0.0, 'o': 0.0}
    n_evals = 0
    for d_idx in range(n_domain):
        for iso in range(8):
            candidate = _evaluate_gene(r_block, all_iso, (d_idx, iso))
            n_evals += 1
            if candidate['mse'] < best['mse']:
                best = candidate
    return best, n_evals


def ga_search_one_range(r_block, all_iso, n_domain,
                        pop_size=40, generation=30, run=1,
                        tournament_size=3, crossover_rate=0.8,
                        mutation_rate=0.15, elitism=2,
                        domain_mutation_radius=8, rng=None):
    """
    Use GA to search the best (domain_idx, isometry) for one range block.
    """
    if rng is None:
        rng = np.random.default_rng()

    pop_size = max(2, int(pop_size))
    generation = max(1, int(generation))
    run = max(1, int(run))
    tournament_size = max(1, int(tournament_size))
    elitism = int(np.clip(elitism, 0, pop_size))
    domain_mutation_radius = max(1, int(domain_mutation_radius))

    best = {'mse': np.inf, 'd_idx': 0, 'iso': 0, 's': 0.0, 'o': 0.0}
    n_evals = 0

    for _ in range(run):
        population = _initial_population(rng, pop_size, n_domain)

        for _gen in range(generation):
            evaluated = [_evaluate_gene(r_block, all_iso, gene) for gene in population]
            n_evals += len(population)
            fitness = np.array([item['mse'] for item in evaluated])

            gen_best_idx = int(np.argmin(fitness))
            if evaluated[gen_best_idx]['mse'] < best['mse']:
                best = evaluated[gen_best_idx]

            elite_idx = np.argsort(fitness)[:elitism]
            next_population = [population[i].copy() for i in elite_idx]

            while len(next_population) < pop_size:
                parent_a = _tournament_select(rng, population, fitness, tournament_size)
                parent_b = _tournament_select(rng, population, fitness, tournament_size)
                child_a, child_b = _crossover(rng, parent_a, parent_b, crossover_rate)
                child_a = _mutate(
                    rng, child_a, n_domain, mutation_rate, domain_mutation_radius
                )
                child_b = _mutate(
                    rng, child_b, n_domain, mutation_rate, domain_mutation_radius
                )
                next_population.append(child_a)
                if len(next_population) < pop_size:
                    next_population.append(child_b)

            population = np.array(next_population, dtype=np.int64)

        evaluated = [_evaluate_gene(r_block, all_iso, gene) for gene in population]
        n_evals += len(population)
        fitness = np.array([item['mse'] for item in evaluated])
        run_best_idx = int(np.argmin(fitness))
        if evaluated[run_best_idx]['mse'] < best['mse']:
            best = evaluated[run_best_idx]

    if not np.isfinite(best['mse']):
        best, fallback_evals = _fallback_valid_candidate(r_block, all_iso, n_domain)
        n_evals += fallback_evals

    return best, n_evals


def encode_ga(image, range_size=8, domain_size=16, domain_stride=8,
              config_path=None, **overrides):
    """
    GA-based FIC Encoder.

    Reads GA parameters from configs/ga.yaml by default. Keyword overrides are
    supported for tests or scripted experiments.
    """
    config = load_ga_config(config_path)
    config.update({k: v for k, v in overrides.items() if v is not None})

    pop_size = int(config['pop_size'])
    generation = int(config['generation'])
    run = int(config['run'])
    tournament_size = int(config['tournament_size'])
    crossover_rate = float(config['crossover_rate'])
    mutation_rate = float(config['mutation_rate'])
    elitism = int(config['elitism'])
    domain_mutation_radius = int(config['domain_mutation_radius'])
    seed = int(config['seed'])

    print("=" * 60)
    print("  GA-based FIC Encoding")
    print("=" * 60)
    print(f"  GA params: pop_size={pop_size}, generation={generation}, "
          f"run={run}, tournament_size={tournament_size}, "
          f"crossover_rate={crossover_rate}, mutation_rate={mutation_rate}, "
          f"elitism={elitism}")

    range_blocks, range_positions = core.extract_range_blocks(image, range_size)
    domain_blocks, domain_positions = core.extract_domain_blocks(
        image, domain_size, domain_stride, range_size
    )
    n_range = len(range_blocks)
    n_domain = len(domain_blocks)

    print(f"  Range blocks: {n_range}, Domain blocks: {n_domain}")
    print(f"  Max evals per range: ~{run * pop_size * (generation + 1)} "
          f"(vs Full Search: {n_domain * 8:,})")
    print()

    print("  Precomputing isometries...", end=" ", flush=True)
    all_iso = core.precompute_all_isometries(domain_blocks)
    print("done")

    print(f"  Running GA for {n_range} range blocks...")
    fractal_codes = []
    total_evals = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for r_idx in range(n_range):
        best, n_evals = ga_search_one_range(
            range_blocks[r_idx],
            all_iso,
            n_domain,
            pop_size=pop_size,
            generation=generation,
            run=run,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=elitism,
            domain_mutation_radius=domain_mutation_radius,
            rng=rng,
        )
        total_evals += n_evals

        fractal_codes.append({
            'range_pos': range_positions[r_idx],
            'domain_idx': best['d_idx'],
            'domain_pos': domain_positions[best['d_idx']],
            'isometry': best['iso'],
            'contrast': best['s'],
            'brightness': best['o'],
            'mse': best['mse'],
        })

        if (r_idx + 1) % 128 == 0 or r_idx == n_range - 1:
            elapsed = time.time() - t0
            pct = (r_idx + 1) / n_range * 100
            eta = elapsed / (r_idx + 1) * (n_range - r_idx - 1)
            avg_evals = total_evals / (r_idx + 1)
            print(f"    [{r_idx+1:4d}/{n_range}] {pct:5.1f}%  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s  "
                  f"avg_evals/block={avg_evals:.0f}")

    encoding_time = time.time() - t0

    all_mse = [c['mse'] for c in fractal_codes]
    mean_mse = float(np.mean(all_mse))
    psnr = 10 * np.log10(255.0 ** 2 / mean_mse) if mean_mse > 0 else float('inf')

    stats = {
        'n_range': n_range,
        'n_domain': n_domain,
        'n_evaluations': total_evals,
        'encoding_time_sec': round(encoding_time, 3),
        'mse_mean': round(mean_mse, 4),
        'mse_max': round(float(np.max(all_mse)), 4),
        'psnr_db': round(psnr, 2),
        'ga_pop_size': pop_size,
        'ga_generation': generation,
        'ga_run': run,
        'ga_tournament_size': tournament_size,
        'ga_crossover_rate': crossover_rate,
        'ga_mutation_rate': mutation_rate,
        'ga_elitism': elitism,
        'ga_domain_mutation_radius': domain_mutation_radius,
    }

    print(f"\n  GA complete | Time: {encoding_time:.2f}s | "
          f"Evals: {total_evals:,} | Avg MSE: {mean_mse:.4f}\n")

    return fractal_codes, encoding_time, stats, domain_positions


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "images/test.png"

    stats = core.run_pipeline(encode_ga, path, method_name='ga')
