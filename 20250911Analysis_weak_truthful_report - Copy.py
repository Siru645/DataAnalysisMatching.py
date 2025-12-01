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
    print(earnings_df[[group_col, 'n_participants', 'mean_earnings', 'std_earnings']])

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


def analyze_overall_earnings(data):
    """Analyze overall earnings for all subjects combined."""
    print(f"\n=== OVERALL AVERAGE EARNINGS (ALL SUBJECTS) ===")

    # Get all payment data, dropping any missing values
    all_payments = data['payment'].dropna()

    if len(all_payments) == 0:
        print("No payment data available.")
        return

    # Calculate summary statistics
    n_participants = len(all_payments)
    mean_earnings = all_payments.mean()
    std_earnings = all_payments.std()

    # Calculate 95% confidence interval for the mean
    sem = stats.sem(all_payments)
    z_score = stats.norm.ppf(0.975)  # 1.96 for 95% CI
    ci_lower = mean_earnings - z_score * sem
    ci_upper = mean_earnings + z_score * sem

    # Display results
    print(f"Total participants: {n_participants}")
    print(f"Mean earnings: ${mean_earnings:.2f}")
    print(f"Standard deviation: ${std_earnings:.2f}")

    return {
        'n_participants': n_participants,
        'mean_earnings': mean_earnings,
        'std_earnings': std_earnings,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,

    }
########################################################################################################################
# VARIABLE GENERATION FUNCTIONS
########################################################################################################################

def generate_weak_truthful_report_variables(data):
    """Generate true_report_r1 to true_report_r20 variables."""
    print("Generating weak truthful report variables...")

    for round_num in range(1, 11):
        # Check xy comparison - wrong if preferences are directly opposite
        xy_correct = ~(
                ((data['preference_1_xy'] == 'X is better') & (
                            data[f'mechanisms_p{round_num}playerxy'] == 'Y is better')) |
                ((data['preference_1_xy'] == 'Y is better') & (
                            data[f'mechanisms_p{round_num}playerxy'] == 'X is better'))
        )

        # Check xz comparison - wrong if preferences are directly opposite
        xz_correct = ~(
                ((data['preference_1_xz'] == 'X is better') & (
                            data[f'mechanisms_p{round_num}playerxz'] == 'Z is better')) |
                ((data['preference_1_xz'] == 'Z is better') & (
                            data[f'mechanisms_p{round_num}playerxz'] == 'X is better'))
        )

        # Check yz comparison - wrong if preferences are directly opposite
        yz_correct = ~(
                ((data['preference_1_yz'] == 'Y is better') & (
                            data[f'mechanisms_p{round_num}playeryz'] == 'Z is better')) |
                ((data['preference_1_yz'] == 'Z is better') & (
                            data[f'mechanisms_p{round_num}playeryz'] == 'Y is better'))
        )

        # All three comparisons must be correct for a true report
        basic_true_report = (xy_correct & xz_correct & yz_correct)
        #Add a special case: if the preference in Part 1 is cyclical, then in Part 2, if irrational_r = 1 it also counts as true
        irrational_exception = (data['preference_1_irrational'] == 1) & (data[f'irrational_r{round_num}'] == 1)
        # Count and print irrational exception cases
        irrational_count = irrational_exception.sum()
        print(f"Round {round_num}: {irrational_count} cases with irrational exception")

        # Combine basic truthfulness with irrational exception
        data[f'weak_true_report_r{round_num}'] = (basic_true_report | irrational_exception).astype(int)

    for round_num in range(11, 21):
        # Check xy comparison - wrong if preferences are directly opposite
        xy_correct = ~(
                ((data['preference_2_xy'] == 'X is better') & (
                            data[f'mechanisms_p{round_num}playerxy'] == 'Y is better')) |
                ((data['preference_2_xy'] == 'Y is better') & (
                            data[f'mechanisms_p{round_num}playerxy'] == 'X is better'))
        )

        # Check xz comparison - wrong if preferences are directly opposite
        xz_correct = ~(
                ((data['preference_2_xz'] == 'X is better') & (
                            data[f'mechanisms_p{round_num}playerxz'] == 'Z is better')) |
                ((data['preference_2_xz'] == 'Z is better') & (
                            data[f'mechanisms_p{round_num}playerxz'] == 'X is better'))
        )

        # Check yz comparison - wrong if preferences are directly opposite
        yz_correct = ~(
                ((data['preference_2_yz'] == 'Y is better') & (
                            data[f'mechanisms_p{round_num}playeryz'] == 'Z is better')) |
                ((data['preference_2_yz'] == 'Z is better') & (
                            data[f'mechanisms_p{round_num}playeryz'] == 'Y is better'))
        )

        # All three comparisons must be correct for a true report
        basic_true_report = (xy_correct & xz_correct & yz_correct)
        #Add a special case: if the preference in Part 1 is cyclical, then in Part 2, if irrational_r = 1 it also counts as true
        irrational_exception = (data['preference_2_irrational'] == 1) & (data[f'irrational_r{round_num}'] == 1)
        # Count and print irrational exception cases
        irrational_count = irrational_exception.sum()
        print(f"Round {round_num}: {irrational_count} cases with irrational exception")

        # Combine basic truthfulness with irrational exception
        data[f'weak_true_report_r{round_num}'] = (basic_true_report | irrational_exception).astype(int)

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
            elif preference_type == "indifferent":
                if r <= 10:
                    filtered_data = treatment_data[treatment_data['incomplete_1'] == 2]
                else:
                    filtered_data = treatment_data[treatment_data['incomplete_2'] == 2]
            elif preference_type == "intransitive":
                if r <= 10:
                    filtered_data = treatment_data[treatment_data['incomplete_1'] == 3]
                else:
                    filtered_data = treatment_data[treatment_data['incomplete_2'] == 3]
            elif preference_type == "other":
                if r <= 10:
                    filtered_data = treatment_data[treatment_data['incomplete_1'] >= 2]
                else:
                    filtered_data = treatment_data[treatment_data['incomplete_2'] >= 2]
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


def subject_level_aggregation_analysis(data, variable_prefix, preference_type="all", rounds_range="11-20",
                                       condition_name="All"):
    """
    Perform subject-level aggregation analysis.
    Calculate each subject's truthful reporting rate across specified rounds,
    then compare between treatments using two-sample tests.
    """
    print(f"\n--- SUBJECT-LEVEL AGGREGATION ANALYSIS (ROUNDS {rounds_range}) - {condition_name.upper()} ---")

    # Determine round range
    if rounds_range == "11-20":
        round_start, round_end = 11, 21
    elif rounds_range == "1-10":
        round_start, round_end = 1, 11

    # Get unique subjects
    subjects = data['participant_code'].unique() if 'participant_code' in data.columns else data.index.unique()

    subject_rates = []

    for subject in subjects:
        subject_data = data[data['participant_code'] == subject] if 'participant_code' in data.columns else data.loc[[subject]]

        if len(subject_data) == 0:
            continue

        # Get treatment for this subject
        treatment = subject_data['treatment'].iloc[0]

        # Apply preference type filtering
        include_subject = True
        if preference_type == "incomplete":
            if rounds_range == "1-10":
                include_subject = subject_data['incomplete_1'].iloc[0] == 1
            else:  # rounds_range == "11-20"
                include_subject = subject_data['incomplete_2'].iloc[0] == 1
        elif preference_type == "complete":
            if rounds_range == "1-10":
                include_subject = subject_data['incomplete_1'].iloc[0] == 0
            else:  # rounds_range == "11-20"
                include_subject = subject_data['incomplete_2'].iloc[0] == 0
        elif preference_type == "indifferent":
            if rounds_range == "1-10":
                include_subject = subject_data['incomplete_1'].iloc[0] == 2
            else:  # rounds_range == "11-20"
                include_subject = subject_data['incomplete_2'].iloc[0] == 2
        elif preference_type == "intransitive":
            if rounds_range == "1-10":
                include_subject = subject_data['incomplete_1'].iloc[0] == 3
            else:  # rounds_range == "11-20"
                include_subject = subject_data['incomplete_2'].iloc[0] == 3

        if not include_subject:
            continue

        # Collect truthful reporting data for this subject across specified rounds
        truthful_reports = []
        valid_rounds = 0

        for round_num in range(round_start, round_end):
            variable_col = f'{variable_prefix}_r{round_num}'
            if variable_col in subject_data.columns:
                value = subject_data[variable_col].iloc[0]
                if pd.notna(value):
                    truthful_reports.append(value)
                    valid_rounds += 1

        # Calculate subject's truthful reporting rate
        if valid_rounds > 0:
            subject_rate = np.mean(truthful_reports)
            subject_rates.append({
                'subject': subject,
                'treatment': treatment,
                'truthful_rate': subject_rate,
                'valid_rounds': valid_rounds,
                'total_truthful': sum(truthful_reports)
            })

    if not subject_rates:
        print("No valid data for subject-level analysis.")
        return None

    # Convert to DataFrame
    subject_df = pd.DataFrame(subject_rates)

    # Separate by treatment
    da_rates = subject_df[subject_df['treatment'] == 'DA']['truthful_rate']
    ttc_rates = subject_df[subject_df['treatment'] == 'TTC']['truthful_rate']

    if len(da_rates) == 0 or len(ttc_rates) == 0:
        print("Insufficient data: need subjects in both treatment groups.")
        return None

    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(
        f"DA  - N: {len(da_rates):3d}, Mean: {da_rates.mean():.4f}, Std: {da_rates.std(ddof=1):.4f}, Min: {da_rates.min():.4f}, Max: {da_rates.max():.4f}")
    print(
        f"TTC - N: {len(ttc_rates):3d}, Mean: {ttc_rates.mean():.4f}, Std: {ttc_rates.std(ddof=1):.4f}, Min: {ttc_rates.min():.4f}, Max: {ttc_rates.max():.4f}")

    # Test for normality (optional, for choosing between parametric/non-parametric tests)
    from scipy.stats import shapiro
    da_normal = shapiro(da_rates)[1] > 0.05 if len(da_rates) >= 3 else True
    ttc_normal = shapiro(ttc_rates)[1] > 0.05 if len(ttc_rates) >= 3 else True

    print(f"\nNORMALITY TESTS:")
    print(f"DA normality (Shapiro-Wilk p > 0.05): {da_normal}")
    print(f"TTC normality (Shapiro-Wilk p > 0.05): {ttc_normal}")

    # Perform tests
    print(f"\nSTATISTICAL TESTS:")

    # 1. Welch's t-test (assumes normality, allows unequal variances)
    t_stat, t_pval = stats.ttest_ind(da_rates, ttc_rates, equal_var=False)
    print(f"Welch's t-test: t = {t_stat:.4f}, p = {t_pval:.4f}, significant = {'Yes' if t_pval < 0.05 else 'No'}")

    # 2. Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(da_rates, ttc_rates, alternative='two-sided')
    print(f"Mann-Whitney U test: U = {u_stat:.4f}, p = {u_pval:.4f}, significant = {'Yes' if u_pval < 0.05 else 'No'}")

    # 3. Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(da_rates) - 1) * da_rates.var(ddof=1) +
                          (len(ttc_rates) - 1) * ttc_rates.var(ddof=1)) /
                         (len(da_rates) + len(ttc_rates) - 2))
    cohens_d = (da_rates.mean() - ttc_rates.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"Effect size (Cohen's d): {cohens_d:.4f}")

    # Confidence intervals for means
    da_ci = stats.t.interval(0.95, len(da_rates) - 1, loc=da_rates.mean(),
                             scale=stats.sem(da_rates)) if len(da_rates) > 1 else (da_rates.mean(), da_rates.mean())
    ttc_ci = stats.t.interval(0.95, len(ttc_rates) - 1, loc=ttc_rates.mean(),
                              scale=stats.sem(ttc_rates)) if len(ttc_rates) > 1 else (
    ttc_rates.mean(), ttc_rates.mean())

    print(f"\nCONFIDENCE INTERVALS (95%):")
    print(f"DA:  [{da_ci[0]:.4f}, {da_ci[1]:.4f}]")
    print(f"TTC: [{ttc_ci[0]:.4f}, {ttc_ci[1]:.4f}]")

    # Save detailed results
    results = {
        'condition': condition_name,
        'rounds': rounds_range,
        'da_n': len(da_rates),
        'ttc_n': len(ttc_rates),
        'da_mean': da_rates.mean(),
        'ttc_mean': ttc_rates.mean(),
        'da_std': da_rates.std(ddof=1),
        'ttc_std': ttc_rates.std(ddof=1),
        'da_ci_lower': da_ci[0],
        'da_ci_upper': da_ci[1],
        'ttc_ci_lower': ttc_ci[0],
        'ttc_ci_upper': ttc_ci[1],
        'welch_t_stat': t_stat,
        'welch_p_value': t_pval,
        'mannwhitney_u_stat': u_stat,
        'mannwhitney_p_value': u_pval,
        'cohens_d': cohens_d,
        'da_normal': da_normal,
        'ttc_normal': ttc_normal
    }

    # Save subject-level data
    filename = f'subject_level_{variable_prefix}_{condition_name.lower().replace(" ", "_")}_{rounds_range.replace("-", "_")}.csv'
    subject_df.to_csv(filename, index=False)
    print(f"\nSubject-level data saved to: {filename}")

    return results, subject_df


def analyze_truthful_report_measure(data, variable_prefix, measure_name, preference_type="all"):
    """Combined function for both pooled and round-by-round analysis."""
    print(f"\n=== {measure_name.upper()} ANALYSIS ({preference_type.upper()}): ROUNDS 11-20 ===")

    # subject_level_aggregation_analysis
    results, subject_data = subject_level_aggregation_analysis(data, variable_prefix, preference_type, "11-20", preference_type.title())

    return results, subject_data



########################################################################################################################
# MAIN ANALYSIS
########################################################################################################################

print("=== TRUTHFUL REPORT ANALYSIS ===")

# 1. Earnings Analysis
print("\n1. EARNINGS ANALYSIS")
analyze_earnings_by_group(data, 'treatment', 'Treatment')
if 'incomplete_pref_pay' in data.columns:
    analyze_earnings_by_group(data, 'incomplete_pref_pay', 'Incomplete Preference Category')
analyze_overall_earnings(data)

# 2. Generate weak truthful report variables
data = generate_weak_truthful_report_variables(data)

# 3. Truthful report analysis
print("\n2. WEAK TRUTHFUL REPORT ANALYSIS")

# 3.1: All subjects
rates_truth_all = calculate_rates_by_treatment(data, 'weak_true_report', "all")

# Calculate sample sizes
da_all_count = len(data[data['treatment'] == 'DA'])
ttc_all_count = len(data[data['treatment'] == 'TTC'])

plot_rates(rates_truth_all,
           'Weak True Report Rates by Treatment: All Subjects',
           'weak_true_report_rates_all_subjects.png',
           'True Report Rate',
           {'rounds_1_10': f'n_TTC:{ttc_all_count}; n_DA:{da_all_count}',
            'rounds_11_20': f'n_TTC:{ttc_all_count}; n_DA:{da_all_count}'})

analyze_truthful_report_measure(data, 'weak_true_report', 'Weak Truthful Report', 'all')

# 3.2: Incomplete preferences
rates_truth_incomplete = calculate_rates_by_treatment(data, 'weak_true_report', "incomplete")

# Calculate sample sizes for incomplete preferences
da_inc_1_10 = len(data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 1)])
ttc_inc_1_10 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 1)])
da_inc_11_20 = len(data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 1)])
ttc_inc_11_20 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 1)])

plot_rates(rates_truth_incomplete,
           'Weak True Report Rates by Treatment: Incomplete Preferences',
           'weak_true_report_rates_incomplete_preferences.png',
           'True Report Rate',
           {'rounds_1_10': f'n_TTC:{ttc_inc_1_10}; n_DA:{da_inc_1_10}',
            'rounds_11_20': f'n_TTC:{ttc_inc_11_20}; n_DA:{da_inc_11_20}'})

analyze_truthful_report_measure(data, 'weak_true_report', 'Weak Truthful Report', 'incomplete')

# 3.3: Complete preferences
rates_truth_complete = calculate_rates_by_treatment(data, 'weak_true_report', "complete")

# Calculate sample sizes for complete preferences
da_comp_1_10 = len(data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 0)])
ttc_comp_1_10 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 0)])
da_comp_11_20 = len(data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 0)])
ttc_comp_11_20 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 0)])

plot_rates(rates_truth_complete,
           'Weak True Report Rates by Treatment: Complete Preferences',
           'weak_true_report_rates_complete_preferences.png',
           'True Report Rate',
           {'rounds_1_10': f'n_TTC:{ttc_comp_1_10}; n_DA:{da_comp_1_10}',
            'rounds_11_20': f'n_TTC:{ttc_comp_11_20}; n_DA:{da_comp_11_20}'})

analyze_truthful_report_measure(data, 'weak_true_report', 'Weak Truthful Report', 'complete')

# 3.4: Other preferences
rates_truth_other = calculate_rates_by_treatment(data, 'weak_true_report', "other")

# Calculate sample sizes for other preferences
da_other_1_10 = len(data[(data['treatment'] == 'DA') & (data['incomplete_1'] >= 2)])
ttc_other_1_10 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_1'] >= 2)])
da_other_11_20 = len(data[(data['treatment'] == 'DA') & (data['incomplete_2'] >= 2)])
ttc_other_11_20 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_2'] >= 2)])

plot_rates(rates_truth_other,
           'Weak True Report Rates by Treatment: Other Preferences',
           'weak_true_report_rates_other_preferences.png',
           'True Report Rate',
           {'rounds_1_10': f'n_TTC:{ttc_other_1_10}; n_DA:{da_other_1_10}',
            'rounds_11_20': f'n_TTC:{ttc_other_11_20}; n_DA:{da_other_11_20}'})

analyze_truthful_report_measure(data, 'weak_true_report', 'Weak Truthful Report', 'other')

# Save the data
data.to_csv('combined_with_weak_truthful_report_analysis.csv', index=False)
print("\nAnalysis complete. Data saved to 'combined_with_weak_truthful_report_analysis.csv'")