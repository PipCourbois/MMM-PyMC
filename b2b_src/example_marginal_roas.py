"""
Example: Using the Enhanced Marginal ROAS Calculation

This script demonstrates how to use the improved calculate_marginal_roas function
with the new prediction_days parameter and visualization.

Key Features:
1. Fixed bug: Each channel's marginal ROAS is calculated independently
2. Only PAID channels: mROAS is calculated only for channels with type='paid' in config
3. New parameter: prediction_days allows focusing on recent periods (e.g., last 30, 60 days)
4. New visualization: plot_marginal_roas_comparison shows spend vs sales vs mROAS

IMPORTANT: Your config file must specify type='paid' for channels where you want mROAS calculated.
Unpaid channels (type='unpaid') are automatically excluded from the analysis.
"""

import yaml
from b2b_src.contributions import calculate_marginal_roas, plot_marginal_roas_comparison
import matplotlib.pyplot as plt

# Assuming you have:
# - fitted_mmm: Your fitted MMM model
# - data: Your dataset with spend and channel data
# - config: Your configuration dictionary

def example_basic_usage(fitted_mmm, data, config):
    """Example 1: Calculate mROAS for the entire dataset"""
    print("=" * 80)
    print("Example 1: Basic mROAS calculation (entire dataset)")
    print("=" * 80)

    mroas_results = calculate_marginal_roas(
        mmm=fitted_mmm,
        data=data,
        config=config,
        marginal_increase=0.01  # 1% spend increase
    )

    # Print results
    print("\nMarginal ROAS Results:")
    for channel, mroas in mroas_results['mroas'].items():
        inc_sales = mroas_results['incremental_sales'][channel]
        inc_spend = mroas_results['incremental_spend'][channel]
        print(f"  {channel}:")
        print(f"    mROAS: {mroas:.2f}")
        print(f"    Incremental Sales: ${inc_sales:,.2f}")
        print(f"    Incremental Spend: ${inc_spend:,.2f}")

    # Create visualization
    fig = plot_marginal_roas_comparison(mroas_results, config)
    plt.savefig('mroas_entire_dataset.png', dpi=150, bbox_inches='tight')
    plt.show()

    return mroas_results


def example_last_30_days(fitted_mmm, data, config):
    """Example 2: Calculate mROAS for the last 30 days"""
    print("\n" + "=" * 80)
    print("Example 2: mROAS calculation for last 30 days")
    print("=" * 80)

    mroas_results = calculate_marginal_roas(
        mmm=fitted_mmm,
        data=data,
        config=config,
        marginal_increase=0.01,  # 1% spend increase
        prediction_days=30  # Focus on last 30 days
    )

    # Print results
    print("\nMarginal ROAS Results (Last 30 Days):")
    period_info = mroas_results['cumulative_period']
    print(f"Period: {period_info['start_date']} to {period_info['end_date']} ({period_info['days']} days)")
    print()

    for channel, mroas in mroas_results['mroas'].items():
        inc_sales = mroas_results['incremental_sales'][channel]
        inc_spend = mroas_results['incremental_spend'][channel]
        print(f"  {channel}:")
        print(f"    mROAS: {mroas:.2f}")
        print(f"    Incremental Sales: ${inc_sales:,.2f}")
        print(f"    Incremental Spend: ${inc_spend:,.2f}")

    # Create visualization
    fig = plot_marginal_roas_comparison(mroas_results, config)
    plt.savefig('mroas_last_30_days.png', dpi=150, bbox_inches='tight')
    plt.show()

    return mroas_results


def example_compare_periods(fitted_mmm, data, config):
    """Example 3: Compare mROAS across different time periods"""
    print("\n" + "=" * 80)
    print("Example 3: Compare mROAS across different periods")
    print("=" * 80)

    periods = [30, 60, 90]
    comparison_results = {}

    for days in periods:
        print(f"\nCalculating mROAS for last {days} days...")
        mroas_results = calculate_marginal_roas(
            mmm=fitted_mmm,
            data=data,
            config=config,
            marginal_increase=0.01,
            prediction_days=days
        )
        comparison_results[days] = mroas_results

    # Print comparison table
    print("\n" + "=" * 80)
    print("Marginal ROAS Comparison Across Periods")
    print("=" * 80)

    # Get channel names from first result
    channels = list(comparison_results[30]['mroas'].keys())

    # Print header
    print(f"\n{'Channel':<25}", end='')
    for days in periods:
        print(f"{'Last ' + str(days) + ' Days':<20}", end='')
    print()
    print("-" * 80)

    # Print each channel's mROAS across periods
    for channel in channels:
        print(f"{channel:<25}", end='')
        for days in periods:
            mroas = comparison_results[days]['mroas'][channel]
            print(f"{mroas:>18.2f}  ", end='')
        print()

    # Create visualizations for each period
    for days in periods:
        fig = plot_marginal_roas_comparison(comparison_results[days], config, figsize=(12, 7))
        plt.savefig(f'mroas_last_{days}_days.png', dpi=150, bbox_inches='tight')
        plt.close()

    return comparison_results


def example_with_date_range(fitted_mmm, data, config):
    """Example 4: Calculate mROAS for a specific date range"""
    print("\n" + "=" * 80)
    print("Example 4: mROAS calculation for specific date range")
    print("=" * 80)

    mroas_results = calculate_marginal_roas(
        mmm=fitted_mmm,
        data=data,
        config=config,
        date_start='2023-01-01',  # Specify start date
        date_end='2023-03-31',    # Specify end date
        marginal_increase=0.01
    )

    # Print results
    print("\nMarginal ROAS Results (Q1 2023):")
    period_info = mroas_results['cumulative_period']
    print(f"Period: {period_info['start_date']} to {period_info['end_date']} ({period_info['days']} days)")
    print()

    for channel, mroas in mroas_results['mroas'].items():
        print(f"  {channel}: mROAS = {mroas:.2f}")

    # Create visualization
    fig = plot_marginal_roas_comparison(mroas_results, config)
    plt.savefig('mroas_q1_2023.png', dpi=150, bbox_inches='tight')
    plt.show()

    return mroas_results


if __name__ == "__main__":
    # Load your configuration
    with open('mmm_config_simu.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Load your data and fitted model
    # data = pd.read_csv(config['data']['source'])
    # fitted_mmm = ... (your fitted MMM model)

    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                     MARGINAL ROAS CALCULATION EXAMPLES                    ║
    ║                                                                           ║
    ║  This script demonstrates the enhanced calculate_marginal_roas function  ║
    ║  with bug fixes and new features.                                        ║
    ║                                                                           ║
    ║  Key Improvements:                                                        ║
    ║  ✓ Fixed: Each channel's mROAS calculated independently                  ║
    ║  ✓ New: prediction_days parameter for recent period analysis             ║
    ║  ✓ New: plot_marginal_roas_comparison visualization                      ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # Uncomment the examples you want to run:

    # Example 1: Basic usage
    # mroas_all = example_basic_usage(fitted_mmm, data, config)

    # Example 2: Last 30 days
    # mroas_30d = example_last_30_days(fitted_mmm, data, config)

    # Example 3: Compare across periods
    # comparison = example_compare_periods(fitted_mmm, data, config)

    # Example 4: Specific date range
    # mroas_q1 = example_with_date_range(fitted_mmm, data, config)

    print("\nNote: Uncomment the examples you want to run in the __main__ section.")
