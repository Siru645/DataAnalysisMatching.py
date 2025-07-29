import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
data = pd.read_csv(
    r'C:\Users\lsr64\OneDrive - Georgia State University\Georgia State University\My Research\Projects\Matching with incomplete preferences\Data analysis\combined.csv')

########################################################################################################################
# UTILITY FUNCTIONS
########################################################################################################################

def calculate_confidence_interval(proportion, n, confidence=0.95):
    """Calculate confidence interval for a proportion with robust error handling."""
    if n <= 0 or pd.isna(proportion) or not (0 <= proportion <= 1):
        return 0, 1
    if n == 1:
        return proportion, proportion

    # Handle edge cases
    if proportion == 0:
        return 0, min(1, 3 / n)  # Rule of three for zero events
    if proportion == 1:
        return max(0, 1 - 3 / n), 1  # Rule of three for all events

    # Use t-distribution for small samples, normal for large samples
    if n >= 30:
        z_score = stats.norm.ppf((1 + confidence) / 2)
        se = np.sqrt(proportion * (1 - proportion) / n)
        margin = z_score * se
    else:
        t_score = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        se = np.sqrt(proportion * (1 - proportion) / n)
        margin = t_score * se

    ci_lower = max(0, proportion - margin)
    ci_upper = min(1, proportion + margin)
    return ci_lower, ci_upper


def analyze_earnings_by_group(data, group_col, group_name):
    """Analyze earnings by any grouping variable."""
    print(f"\n=== AVERAGE EARNINGS BY {group_name.upper()} ===")

    earnings_detailed = []
    for group_val in sorted(data[group_col].unique()):
        if pd.notna(group_val):
            group_data = data[data[group_col] == group_val]['payment'].dropna()
            if len(group_data) > 0:
                mean_earnings = group_data.mean()
                n = len(group_data)
                sem = stats.sem(group_data)
                z_score = stats.norm.ppf(0.975)  # 1.96 for 95% CI
                ci_lower = mean_earnings - z_score * sem
                ci_upper = mean_earnings + z_score * sem

                earnings_detailed.append({
                    group_col: group_val,
                    'n_participants': n,
                    'mean_earnings': mean_earnings,
                    'std_earnings': group_data.std(),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })

    earnings_df = pd.DataFrame(earnings_detailed).round(2)
    earnings_df['ci_range'] = earnings_df.apply(lambda x: f"[{x['ci_lower']}, {x['ci_upper']}]", axis=1)
    print(earnings_df[[group_col, 'n_participants', 'mean_earnings', 'ci_range', 'std_earnings']])

    # Statistical tests
    groups = [data[data[group_col] == val]['payment'].dropna()
              for val in sorted(data[group_col].unique()) if pd.notna(val)]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) == 2:
        t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
        print(f"\nt-test: t={t_stat:.4f}, p={p_value:.4f}, significant={'Yes' if p_value < 0.05 else 'No'}")
    elif len(groups) > 2:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"\nANOVA: F={f_stat:.4f}, p={p_value:.4f}, significant={'Yes' if p_value < 0.05 else 'No'}")


########################################################################################################################
# VARIABLE GENERATION FUNCTIONS
########################################################################################################################

def generate_truthful_report_variables(data):
    """Generate true_report_r1 to true_report_r20 variables."""
    print("Generating truthful report variables...")

    for round_num in range(1, 11):
        data[f'true_report_r{round_num}'] = (
                (data['preference_1_xy'] == data[f'mechanisms_p{round_num}playerxy']) &
                (data['preference_1_xz'] == data[f'mechanisms_p{round_num}playerxz']) &
                (data['preference_1_yz'] == data[f'mechanisms_p{round_num}playeryz'])
        ).astype(int)

    for round_num in range(11, 21):
        data[f'true_report_r{round_num}'] = (
                (data['preference_2_xy'] == data[f'mechanisms_p{round_num}playerxy']) &
                (data['preference_2_xz'] == data[f'mechanisms_p{round_num}playerxz']) &
                (data['preference_2_yz'] == data[f'mechanisms_p{round_num}playeryz'])
        ).astype(int)

    return data


########################################################################################################################
# ANALYSIS FUNCTIONS
########################################################################################################################

def calculate_rates_by_treatment(data, variable_prefix, preference_type="all"):
    """Calculate rates by treatment with proper period-specific filtering."""
    rates_data = []
    treatments = ['DA', 'TTC']

    for treatment in treatments:
        treatment_data = data[data['treatment'] == treatment]

        for r in range(1, 21):
            # Apply proper filtering based on round and preference type
            if preference_type == "incomplete":
                if r <= 10:
                    filtered_data = treatment_data[treatment_data['incomplete_1'] == 1]
                else:
                    filtered_data = treatment_data[treatment_data['incomplete_2'] == 1]
            elif preference_type == "complete":
                if r <= 10:
                    filtered_data = treatment_data[treatment_data['incomplete_1'] == 0]
                else:
                    filtered_data = treatment_data[treatment_data['incomplete_2'] == 0]
            elif preference_type == "other":
                if r <= 10:
                    filtered_data = treatment_data[treatment_data['incomplete_1'] == 2]
                else:
                    filtered_data = treatment_data[treatment_data['incomplete_2'] == 2]
            else:  # "all"
                filtered_data = treatment_data

            # Get the rate data for this round
            round_data = filtered_data[f'{variable_prefix}_r{r}'].dropna()

            if len(round_data) > 0:
                rate = round_data.mean()
                n = len(round_data)
                ci_lower, ci_upper = calculate_confidence_interval(rate, n)

                rates_data.append({
                    'treatment': treatment,
                    'round': r,
                    'rate': rate,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n': n,
                    'round_group': "Rounds 1-10" if r <= 10 else "Rounds 11-20"
                })
            else:
                rates_data.append({
                    'treatment': treatment,
                    'round': r,
                    'rate': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n': 0,
                    'round_group': "Rounds 1-10" if r <= 10 else "Rounds 11-20"
                })

    return pd.DataFrame(rates_data)


def plot_rates(rates_df, title, filename, ylabel, sample_info):
    """Generic plotting function for rates."""
    plt.figure(figsize=(14, 8))

    for treatment, color, marker in [('DA', 'red', 's-'), ('TTC', 'blue', 'o-')]:
        treatment_data = rates_df[rates_df['treatment'] == treatment].copy()
        if not treatment_data.empty:
            valid_data = treatment_data.dropna(subset=['rate', 'ci_lower', 'ci_upper'])

            if not valid_data.empty:
                lower_errors = valid_data['rate'] - valid_data['ci_lower']
                upper_errors = valid_data['ci_upper'] - valid_data['rate']

                plt.errorbar(valid_data['round'], valid_data['rate'],
                             yerr=[lower_errors, upper_errors],
                             fmt=marker, color=color, label=treatment, capsize=3, linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel('Round', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(range(1, 21))
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=10.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    plt.text(5.5, 0.95, f'Rounds 1-10\n{sample_info["rounds_1_10"]}', ha='center', fontsize=10, alpha=0.7)
    plt.text(15.5, 0.95, f'Rounds 11-20\n{sample_info["rounds_11_20"]}', ha='center', fontsize=10, alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def pooled_analysis(data, variable_prefix, preference_type="all", rounds_range="11-20", condition_name="All"):
    """Perform pooled analysis for specified rounds and preference type."""
    print(f"\n--- POOLED ANALYSIS (ROUNDS {rounds_range}) - {condition_name.upper()} ---")

    # Determine round range
    if rounds_range == "11-20":
        round_start, round_end = 11, 21
    elif rounds_range == "1-10":
        round_start, round_end = 1, 11
    else:
        round_start, round_end = 1, 21

    ttc_rates_pooled = []
    da_rates_pooled = []

    for round_num in range(round_start, round_end):
        # Apply proper filtering
        if preference_type == "incomplete":
            if round_num <= 10:
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 1)]
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 1)]
            else:
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 1)]
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 1)]
        elif preference_type == "complete":
            if round_num <= 10:
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 0)]
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 0)]
            else:
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 0)]
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 0)]
        elif preference_type == "other":
            if round_num <= 10:
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 2)]
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 2)]
            else:
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 2)]
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 2)]
        else:  # "all"
            ttc_data = data[data['treatment'] == 'TTC']
            da_data = data[data['treatment'] == 'DA']

        variable_col = f'{variable_prefix}_r{round_num}'
        ttc_round = ttc_data[variable_col].dropna() if variable_col in ttc_data.columns else pd.Series(dtype=float)
        da_round = da_data[variable_col].dropna() if variable_col in da_data.columns else pd.Series(dtype=float)

        if len(ttc_round) > 0:
            ttc_rates_pooled.extend(ttc_round.tolist())
        if len(da_round) > 0:
            da_rates_pooled.extend(da_round.tolist())

    if len(ttc_rates_pooled) > 0 and len(da_rates_pooled) > 0:
        ttc_rates_pooled = np.array(ttc_rates_pooled)
        da_rates_pooled = np.array(da_rates_pooled)

        print(
            f"DA  - Mean: {da_rates_pooled.mean():.4f}, Std: {da_rates_pooled.std(ddof=1):.4f}, N: {len(da_rates_pooled)}")
        print(
            f"TTC - Mean: {ttc_rates_pooled.mean():.4f}, Std: {ttc_rates_pooled.std(ddof=1):.4f}, N: {len(ttc_rates_pooled)}")

        t_stat, p_value = stats.ttest_ind(da_rates_pooled, ttc_rates_pooled)
        print(f"\nPooled t-test: t={t_stat:.4f}, p={p_value:.4f}, significant={'Yes' if p_value < 0.05 else 'No'}")

        return {
            'da_mean': da_rates_pooled.mean(),
            'ttc_mean': ttc_rates_pooled.mean(),
            'da_n': len(da_rates_pooled),
            'ttc_n': len(ttc_rates_pooled),
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    else:
        print("Insufficient data for pooled analysis.")
        return None


def round_by_round_analysis(data, variable_prefix, preference_type="all", rounds_range="11-20", condition_name="All"):
    """Perform round-by-round analysis."""
    print(f"\n--- ROUND-BY-ROUND ANALYSIS (ROUNDS {rounds_range}) - {condition_name.upper()} ---")

    # Determine round range
    if rounds_range == "11-20":
        round_start, round_end = 11, 21
    elif rounds_range == "1-10":
        round_start, round_end = 1, 11
    else:
        round_start, round_end = 1, 21

    round_results = []
    for round_num in range(round_start, round_end):
        # Apply proper filtering
        if preference_type == "incomplete":
            if round_num <= 10:
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 1)]
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 1)]
            else:
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 1)]
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 1)]
        elif preference_type == "complete":
            if round_num <= 10:
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 0)]
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 0)]
            else:
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 0)]
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 0)]
        elif preference_type == "other":
            if round_num <= 10:
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 2)]
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 2)]
            else:
                da_data = data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 2)]
                ttc_data = data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 2)]
        else:  # "all"
            da_data = data[data['treatment'] == 'DA']
            ttc_data = data[data['treatment'] == 'TTC']

        variable_col = f'{variable_prefix}_r{round_num}'
        da_round = da_data[variable_col].dropna() if variable_col in da_data.columns else pd.Series(dtype=float)
        ttc_round = ttc_data[variable_col].dropna() if variable_col in ttc_data.columns else pd.Series(dtype=float)

        if len(da_round) > 0 and len(ttc_round) > 0:
            t_stat, p_val = stats.ttest_ind(da_round, ttc_round)
            round_results.append({
                'round': round_num,
                'da_mean': da_round.mean(),
                'ttc_mean': ttc_round.mean(),
                'da_n': len(da_round),
                'ttc_n': len(ttc_round),
                't_stat': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
        else:
            round_results.append({
                'round': round_num,
                'da_mean': da_round.mean() if len(da_round) > 0 else np.nan,
                'ttc_mean': ttc_round.mean() if len(ttc_round) > 0 else np.nan,
                'da_n': len(da_round),
                'ttc_n': len(ttc_round),
                't_stat': np.nan,
                'p_value': np.nan,
                'significant': False
            })

    if round_results:
        round_df = pd.DataFrame(round_results)
        print(round_df[['round', 'da_mean', 'ttc_mean', 'da_n', 'ttc_n', 't_stat', 'p_value', 'significant']].round(4))

        # Save to CSV
        filename = f'round_by_round_{variable_prefix}_{condition_name.lower().replace(" ", "_")}_{rounds_range.replace("-", "_")}.csv'
        round_df.to_csv(filename, index=False)

        # Summary
        significant_rounds = round_df['significant'].sum()
        total_testable_rounds = round_df['p_value'].notna().sum()
        if total_testable_rounds > 0:
            print(f"\nSummary: {significant_rounds} out of {total_testable_rounds} rounds show significant differences")

        return round_df
    else:
        print("No data available for round-by-round analysis.")
        return None


def analyze_truthful_report_measure(data, variable_prefix, measure_name, preference_type="all"):
    """Combined function for both pooled and round-by-round analysis."""
    print(f"\n=== {measure_name.upper()} ANALYSIS ({preference_type.upper()}): ROUNDS 11-20 ===")

    # Pooled analysis
    pooled_results = pooled_analysis(data, variable_prefix, preference_type, "11-20", preference_type.title())

    # Round-by-round analysis
    round_results = round_by_round_analysis(data, variable_prefix, preference_type, "11-20", preference_type.title())

    return pooled_results, round_results


########################################################################################################################
# MAIN ANALYSIS
########################################################################################################################

print("=== TRUTHFUL REPORT ANALYSIS ===")

# 1. Earnings Analysis
print("\n1. EARNINGS ANALYSIS")
analyze_earnings_by_group(data, 'treatment', 'Treatment')
if 'incomplete_pref_pay' in data.columns:
    analyze_earnings_by_group(data, 'incomplete_pref_pay', 'Incomplete Preference Category')

# 2. Generate truthful report variables
data = generate_truthful_report_variables(data)

# 3. Truthful report analysis
print("\n2. TRUTHFUL REPORT ANALYSIS")

# 3.1: All subjects
rates_truth_all = calculate_rates_by_treatment(data, 'true_report', "all")

# Calculate sample sizes
da_all_count = len(data[data['treatment'] == 'DA'])
ttc_all_count = len(data[data['treatment'] == 'TTC'])

plot_rates(rates_truth_all,
           'True Report Rates by Treatment: All Subjects',
           'true_report_rates_all_subjects.png',
           'True Report Rate',
           {'rounds_1_10': f'n_TTC:{ttc_all_count}; n_DA:{da_all_count}',
            'rounds_11_20': f'n_TTC:{ttc_all_count}; n_DA:{da_all_count}'})

analyze_truthful_report_measure(data, 'true_report', 'Truthful Report', 'all')

# 3.2: Incomplete preferences
rates_truth_incomplete = calculate_rates_by_treatment(data, 'true_report', "incomplete")

# Calculate sample sizes for incomplete preferences
da_inc_1_10 = len(data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 1)])
ttc_inc_1_10 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 1)])
da_inc_11_20 = len(data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 1)])
ttc_inc_11_20 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 1)])

plot_rates(rates_truth_incomplete,
           'True Report Rates by Treatment: Incomplete Preferences',
           'true_report_rates_incomplete_preferences.png',
           'True Report Rate',
           {'rounds_1_10': f'n_TTC:{ttc_inc_1_10}; n_DA:{da_inc_1_10}',
            'rounds_11_20': f'n_TTC:{ttc_inc_11_20}; n_DA:{da_inc_11_20}'})

analyze_truthful_report_measure(data, 'true_report', 'Truthful Report', 'incomplete')

# 3.3: Complete preferences
rates_truth_complete = calculate_rates_by_treatment(data, 'true_report', "complete")

# Calculate sample sizes for complete preferences
da_comp_1_10 = len(data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 0)])
ttc_comp_1_10 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 0)])
da_comp_11_20 = len(data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 0)])
ttc_comp_11_20 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 0)])

plot_rates(rates_truth_complete,
           'True Report Rates by Treatment: Complete Preferences',
           'true_report_rates_complete_preferences.png',
           'True Report Rate',
           {'rounds_1_10': f'n_TTC:{ttc_comp_1_10}; n_DA:{da_comp_1_10}',
            'rounds_11_20': f'n_TTC:{ttc_comp_11_20}; n_DA:{da_comp_11_20}'})

analyze_truthful_report_measure(data, 'true_report', 'Truthful Report', 'complete')

# 3.4: Other preferences
rates_truth_other = calculate_rates_by_treatment(data, 'true_report', "other")

# Calculate sample sizes for other preferences
da_other_1_10 = len(data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 2)])
ttc_other_1_10 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 2)])
da_other_11_20 = len(data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 2)])
ttc_other_11_20 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 2)])

plot_rates(rates_truth_other,
           'True Report Rates by Treatment: Other Preferences',
           'true_report_rates_other_preferences.png',
           'True Report Rate',
           {'rounds_1_10': f'n_TTC:{ttc_other_1_10}; n_DA:{da_other_1_10}',
            'rounds_11_20': f'n_TTC:{ttc_other_11_20}; n_DA:{da_other_11_20}'})

analyze_truthful_report_measure(data, 'true_report', 'Truthful Report', 'other')

# Save the data
data.to_csv('combined_with_truthful_report_analysis.csv', index=False)
print("\nAnalysis complete. Data saved to 'combined_with_truthful_report_analysis.csv'")