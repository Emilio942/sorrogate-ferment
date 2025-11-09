"""
Benchmark script for HF model: estimate budget requirements.
Measures execution time and estimates max episodes within budget.
"""
import argparse
import sys
from pathlib import Path
import json

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import HF_PARAMS, EPISODE_HORIZON, HF_BUDGET_SECONDS, get_results_path
from hf_model import benchmark_episode_time
from utils import format_time


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark HF model execution time'
    )
    parser.add_argument(
        '--n_episodes',
        type=int,
        default=10,
        help='Number of episodes for benchmarking (default: 10)'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=EPISODE_HORIZON,
        help=f'Episode horizon (default: {EPISODE_HORIZON})'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for benchmark results JSON'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HF MODEL BENCHMARK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Horizon: {args.horizon} steps")
    print(f"  Budget: {format_time(HF_BUDGET_SECONDS)}")
    
    # Run benchmark
    stats = benchmark_episode_time(
        n_episodes=args.n_episodes,
        horizon=args.horizon
    )
    
    # Save results
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = get_results_path('hf_benchmark')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n💾 Benchmark results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
