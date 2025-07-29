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


def is_indifferent(pref_lists):
    """Check if a student has indifferent preferences."""
    return pref_lists == "Indifference"


def get_school_priority_order():
    """Define the priority orders for each school."""
    return {
        'X': ['a', 'b', 'c'],
        'Y': ['b', 'a', 'c'],
        'Z': ['c', 'b', 'a']
    }


def student_prefers_school_over_current(student_pref, preferred_school, current_school):
    """Check if student prefers preferred_school over current_school in ALL rankings."""
    if is_indifferent(student_pref):
        return True

    if preferred_school == current_school:
        return False

    rankings = student_pref.split(';')

    for ranking in rankings:
        choices = ranking.split(',')
        preferred_pos = choices.index(preferred_school)
        current_pos = choices.index(current_school)

        if preferred_pos >= current_pos:
            return False

    return True


def school_prefers_student_over_current(school, preferred_student_role, current_student_role):
    """Check if school prefers preferred_student over current_student based on priority order."""
    school_priorities = get_school_priority_order()
    priority_order = school_priorities[school]

    preferred_pos = priority_order.index(preferred_student_role)
    current_pos = priority_order.index(current_student_role)

    return preferred_pos < current_pos


########################################################################################################################
# VARIABLE GENERATION FUNCTIONS
########################################################################################################################

def generate_truthful_report_variables(data):
    """Generate true_report_r1 to true_report_r20 variables."""
    print("Generating truthful report variables...")

    # Create all columns at once to avoid fragmentation
    new_columns = {}

    for round_num in range(1, 11):
        new_columns[f'true_report_r{round_num}'] = (
                (data['preference_1_xy'] == data[f'mechanisms_p{round_num}playerxy']) &
                (data['preference_1_xz'] == data[f'mechanisms_p{round_num}playerxz']) &
                (data['preference_1_yz'] == data[f'mechanisms_p{round_num}playeryz'])
        ).astype(int)

    for round_num in range(11, 21):
        new_columns[f'true_report_r{round_num}'] = (
                (data['preference_2_xy'] == data[f'mechanisms_p{round_num}playerxy']) &
                (data['preference_2_xz'] == data[f'mechanisms_p{round_num}playerxz']) &
                (data['preference_2_yz'] == data[f'mechanisms_p{round_num}playeryz'])
        ).astype(int)

    # Add all new columns at once
    new_df = pd.DataFrame(new_columns, index=data.index)
    data = pd.concat([data, new_df], axis=1)

    return data


def generate_subsample_2truth_variables(data):
    """Generate subsample_2truth_r1 to subsample_2truth_r20 variables."""
    print("Generating subsample 2-truth variables...")

    # Initialize all subsample columns at once
    subsample_columns = {}
    for round_num in range(1, 21):
        subsample_columns[f'subsample_2truth_r{round_num}'] = 0

    # Add all columns at once
    subsample_df = pd.DataFrame(subsample_columns, index=data.index)
    data = pd.concat([data, subsample_df], axis=1)

    for round_num in range(1, 21):
        group_col = f'group_id_r{round_num}'
        truth_col = f'true_report_r{round_num}'

        if group_col not in data.columns or truth_col not in data.columns:
            continue

        for group_id, group_data in data.groupby(group_col):
            if pd.isna(group_id) or len(group_data) != 3:
                continue

            # Count truthful reports in this group
            truth_count = group_data[truth_col].sum()

            # If at least 2 subjects truthfully report, mark all subjects in this group
            if truth_count >= 2:
                group_indices = group_data.index.tolist()
                data.loc[group_indices, f'subsample_2truth_r{round_num}'] = 1

    return data


def generate_blocking_pair_variables(data):
    """Generate block_r1 to block_r20 variables."""
    print("Generating blocking pair variables...")

    # Initialize all blocking columns at once
    blocking_columns = {}
    for round_num in range(1, 21):
        blocking_columns[f'block_r{round_num}'] = 0

    # Add all columns at once
    blocking_df = pd.DataFrame(blocking_columns, index=data.index)
    data = pd.concat([data, blocking_df], axis=1)

    for round_num in range(1, 21):
        group_col = f'group_id_r{round_num}'
        assignment_col = f'assigned_school_r{round_num}'
        role_col = f'role_r{round_num}'
        pref_col = 'preference_1_lists' if round_num <= 10 else 'preference_2_lists'

        if group_col not in data.columns:
            continue

        for group_id, group_data in data.groupby(group_col):
            if pd.isna(group_id) or len(group_data) != 3:
                continue

            # Store original indices and reset group data
            original_indices = group_data.index.tolist()
            group_df = group_data.reset_index(drop=True)

            # Check required columns
            required_cols = [assignment_col, role_col, pref_col]
            if not all(col in group_df.columns for col in required_cols):
                continue

            # Get student data
            students = []
            for _, student in group_df.iterrows():
                students.append({
                    'role': student[role_col],
                    'assignment': student[assignment_col],
                    'preferences': student[pref_col]
                })

            students_in_blocking_pairs = set()
            schools = ['X', 'Y', 'Z']

            # Check all possible blocking pairs
            for school in schools:
                # Find currently assigned student
                current_student_idx = None
                for i, student in enumerate(students):
                    if student['assignment'] == school:
                        current_student_idx = i
                        break

                if current_student_idx is None:
                    continue

                current_student = students[current_student_idx]

                # Check other students
                for i, other_student in enumerate(students):
                    if i == current_student_idx:
                        continue

                    # Check blocking pair conditions
                    student_prefers = student_prefers_school_over_current(
                        other_student['preferences'],
                        school,
                        other_student['assignment']
                    )

                    school_prefers = school_prefers_student_over_current(
                        school,
                        other_student['role'],
                        current_student['role']
                    )

                    if student_prefers and school_prefers:
                        students_in_blocking_pairs.add(i)

            # Mark students in blocking pairs
            if students_in_blocking_pairs:
                for student_idx in students_in_blocking_pairs:
                    if student_idx < len(original_indices):
                        actual_index = original_indices[student_idx]
                        data.loc[actual_index, f'block_r{round_num}'] = 1

    return data

def generate_envy_free_variables(data):
    """Generate envy_free_r1 to envy_free_r20 variables."""
    print("Generating envy-free variables...")

    # Prepare all new columns at once
    envy_free_columns = {}

    for round_num in range(1, 21):
        block_col = f'block_r{round_num}'
        envy_free_col = f'envy_free_r{round_num}'

        if block_col in data.columns:
            # envy_free is the inverse of block
            envy_free_columns[envy_free_col] = 1 - data[block_col]
        else:
            envy_free_columns[envy_free_col] = 0

    # Add envy_free columns
    envy_free_df = pd.DataFrame(envy_free_columns, index=data.index)
    data = pd.concat([data, envy_free_df], axis=1)

    return data


def generate_stability_variables(data):
    """Generate count_block and stable_group variables."""
    print("Generating stability variables...")

    # Prepare all new columns at once
    count_block_columns = {}
    stable_columns = {}

    for round_num in range(1, 21):
        group_col = f'group_id_r{round_num}'
        block_col = f'block_r{round_num}'
        count_block_col = f'count_block_r{round_num}'

        if group_col not in data.columns or block_col not in data.columns:
            count_block_columns[count_block_col] = 0
        else:
            # Calculate count of students with block=1 for each group
            group_block_counts = data.groupby(group_col)[block_col].sum()
            count_block_columns[count_block_col] = data[group_col].map(group_block_counts).fillna(0)

    # Add count_block columns
    count_block_df = pd.DataFrame(count_block_columns, index=data.index)
    data = pd.concat([data, count_block_df], axis=1)

    # Generate stable_group variables
    for round_num in range(1, 21):
        count_block_col = f'count_block_r{round_num}'
        stable_col = f'stable_group_r{round_num}'

        if count_block_col in data.columns:
            stable_columns[stable_col] = (data[count_block_col] == 0).astype(int)
        else:
            stable_columns[stable_col] = 0

    # Add stable_group columns
    stable_df = pd.DataFrame(stable_columns, index=data.index)
    data = pd.concat([data, stable_df], axis=1)

    return data


########################################################################################################################
# ANALYSIS FUNCTIONS
########################################################################################################################

def calculate_rates_by_treatment(data, variable_prefix, preference_type="all", subsample_filter=False):
    """Calculate rates by treatment with proper period-specific filtering."""
    rates_data = []
    treatments = ['DA', 'TTC']

    for treatment in treatments:
        treatment_data = data[data['treatment'] == treatment]

        for r in range(1, 21):
            # Start with fresh treatment data for each round (FIXED)
            round_treatment_data = treatment_data.copy()

            # Apply subsample filter for THIS SPECIFIC ROUND only
            if subsample_filter:
                subsample_col = f'subsample_2truth_r{r}'
                if subsample_col in round_treatment_data.columns:
                    round_treatment_data = round_treatment_data[round_treatment_data[subsample_col] == 1]
                else:
                    # If subsample column doesn't exist, skip this round
                    rates_data.append({
                        'treatment': treatment,
                        'round': r,
                        'rate': np.nan,
                        'ci_lower': np.nan,
                        'ci_upper': np.nan,
                        'n': 0,
                        'round_group': "Rounds 1-10" if r <= 10 else "Rounds 11-20"
                    })
                    continue

            # Apply proper filtering based on round and preference type
            if preference_type == "incomplete":
                if r <= 10:
                    filtered_data = round_treatment_data[round_treatment_data['incomplete_1'] == 1]
                else:
                    filtered_data = round_treatment_data[round_treatment_data['incomplete_2'] == 1]
            elif preference_type == "complete":
                if r <= 10:
                    filtered_data = round_treatment_data[round_treatment_data['incomplete_1'] == 0]
                else:
                    filtered_data = round_treatment_data[round_treatment_data['incomplete_2'] == 0]
            elif preference_type == "other":
                if r <= 10:
                    filtered_data = round_treatment_data[round_treatment_data['incomplete_1'] == 2]
                else:
                    filtered_data = round_treatment_data[round_treatment_data['incomplete_2'] == 2]
            else:  # "all"
                filtered_data = round_treatment_data

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


def calculate_group_stability_rates(data, subsample_filter=False):
    """Calculate group stability rates by treatment for all subjects."""
    rates_data = []
    treatments = ['DA', 'TTC']

    for treatment in treatments:
        treatment_data = data[data['treatment'] == treatment]

        for round_num in range(1, 21):
            # Start with fresh treatment data for each round (FIXED)
            round_treatment_data = treatment_data.copy()

            # Apply subsample filter for THIS SPECIFIC ROUND only
            if subsample_filter:
                subsample_col = f'subsample_2truth_r{round_num}'
                if subsample_col in round_treatment_data.columns:
                    treatment_data_filtered = round_treatment_data[round_treatment_data[subsample_col] == 1]
                else:
                    rates_data.append({
                        'treatment': treatment,
                        'round': round_num,
                        'stability_rate': np.nan,
                        'ci_lower': np.nan,
                        'ci_upper': np.nan,
                        'n': 0,
                        'round_group': "Rounds 1-10" if round_num <= 10 else "Rounds 11-20"
                    })
                    continue
            else:
                treatment_data_filtered = round_treatment_data

            group_col = f'group_id_r{round_num}'
            stable_col = f'stable_group_r{round_num}'

            if group_col not in treatment_data_filtered.columns or stable_col not in treatment_data_filtered.columns:
                rates_data.append({
                    'treatment': treatment,
                    'round': round_num,
                    'stability_rate': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n': 0,
                    'round_group': "Rounds 1-10" if round_num <= 10 else "Rounds 11-20"
                })
                continue

            # Get one record per group
            group_stability = treatment_data_filtered.groupby(group_col)[stable_col].first().dropna()

            if len(group_stability) > 0:
                stability_rate = group_stability.mean()
                n = len(group_stability)
                ci_lower, ci_upper = calculate_confidence_interval(stability_rate, n)

                rates_data.append({
                    'treatment': treatment,
                    'round': round_num,
                    'stability_rate': stability_rate,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n': n,
                    'round_group': "Rounds 1-10" if round_num <= 10 else "Rounds 11-20"
                })
            else:
                rates_data.append({
                    'treatment': treatment,
                    'round': round_num,
                    'stability_rate': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n': 0,
                    'round_group': "Rounds 1-10" if round_num <= 10 else "Rounds 11-20"
                })

    return pd.DataFrame(rates_data)


def plot_rates(rates_df, title, filename, ylabel, sample_info):
    """Generic plotting function for rates."""
    plt.figure(figsize=(14, 8))

    for treatment, color, marker in [('DA', 'red', 's-'), ('TTC', 'blue', 'o-')]:
        treatment_data = rates_df[rates_df['treatment'] == treatment].copy()
        if not treatment_data.empty:
            valid_data = treatment_data.dropna(
                subset=['rate' if 'rate' in treatment_data.columns else 'stability_rate',
                        'ci_lower', 'ci_upper'])

            if not valid_data.empty:
                rate_col = 'rate' if 'rate' in valid_data.columns else 'stability_rate'
                lower_errors = valid_data[rate_col] - valid_data['ci_lower']
                upper_errors = valid_data['ci_upper'] - valid_data[rate_col]

                plt.errorbar(valid_data['round'], valid_data[rate_col],
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


def pooled_analysis(data, variable_prefix, preference_type="all", rounds_range="11-20", condition_name="All",
                    is_group_level=False, subsample_filter=False):
    """Perform pooled analysis for specified rounds and preference type."""
    subsample_suffix = " (Subsample: ≥2 Truthful)" if subsample_filter else ""
    print(f"\n--- POOLED ANALYSIS (ROUNDS {rounds_range}) - {condition_name.upper()}{subsample_suffix} ---")

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

        # Apply subsample filter if requested
        if subsample_filter:
            subsample_col = f'subsample_2truth_r{round_num}'
            if subsample_col in ttc_data.columns:
                ttc_data = ttc_data[ttc_data[subsample_col] == 1]
                da_data = da_data[da_data[subsample_col] == 1]

        # Handle group-level analysis differently
        if is_group_level:
            group_col = f'group_id_r{round_num}'
            variable_col = f'{variable_prefix}_r{round_num}'

            if group_col in ttc_data.columns and variable_col in ttc_data.columns:
                ttc_round = ttc_data.groupby(group_col)[variable_col].first().dropna()
                da_round = da_data.groupby(group_col)[variable_col].first().dropna()
            else:
                ttc_round = pd.Series(dtype=float)
                da_round = pd.Series(dtype=float)
        else:
            variable_col = f'{variable_prefix}_r{round_num}'
            ttc_round = ttc_data[variable_col].dropna() if variable_col in ttc_data.columns else pd.Series(dtype=float)
            da_round = da_data[variable_col].dropna() if variable_col in da_data.columns else pd.Series(dtype=float)

        # Pool data across rounds (this is the key difference from round_by_round_analysis)
        if len(ttc_round) > 0:
            ttc_rates_pooled.extend(ttc_round.tolist())
        if len(da_round) > 0:
            da_rates_pooled.extend(da_round.tolist())

    # Perform pooled statistical analysis
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


def round_by_round_analysis(data, variable_prefix, preference_type="all", rounds_range="11-20", condition_name="All",
                            is_group_level=False, subsample_filter=False):
    """Perform round-by-round analysis."""
    subsample_suffix = " (Subsample: ≥2 Truthful)" if subsample_filter else ""
    print(f"\n--- ROUND-BY-ROUND ANALYSIS (ROUNDS {rounds_range}) - {condition_name.upper()}{subsample_suffix} ---")

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

        # Apply subsample filter if requested
        if subsample_filter:
            subsample_col = f'subsample_2truth_r{round_num}'
            if subsample_col in da_data.columns:
                da_data = da_data[da_data[subsample_col] == 1]
                ttc_data = ttc_data[ttc_data[subsample_col] == 1]

        # Handle group-level analysis differently
        if is_group_level:
            group_col = f'group_id_r{round_num}'
            variable_col = f'{variable_prefix}_r{round_num}'

            if group_col in da_data.columns and variable_col in da_data.columns:
                da_round = da_data.groupby(group_col)[variable_col].first().dropna()
                ttc_round = ttc_data.groupby(group_col)[variable_col].first().dropna()
            else:
                da_round = pd.Series(dtype=float)
                ttc_round = pd.Series(dtype=float)
        else:
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
        level_suffix = "group_level" if is_group_level else "individual_level"
        subsample_suffix = "_subsample_2truth" if subsample_filter else ""
        filename = f'round_by_round_{variable_prefix}_{condition_name.lower().replace(" ", "_")}_{level_suffix}{subsample_suffix}_{rounds_range.replace("-", "_")}.csv'
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


def analyze_stability_measure(data, variable_prefix, measure_name, preference_type="all", is_group_level=False,
                              subsample_filter=False):
    """Combined function for both pooled and round-by-round analysis."""
    subsample_suffix = " (Subsample: ≥2 Truthful)" if subsample_filter else ""
    print(f"\n=== {measure_name.upper()} ANALYSIS ({preference_type.upper()}): ROUNDS 11-20{subsample_suffix} ===")

    # Pooled analysis
    pooled_results = pooled_analysis(data, variable_prefix, preference_type, "11-20", preference_type.title(),
                                     is_group_level, subsample_filter)

    # Round-by-round analysis
    round_results = round_by_round_analysis(data, variable_prefix, preference_type, "11-20", preference_type.title(),
                                            is_group_level, subsample_filter)

    return pooled_results, round_results


def calculate_subsample_sizes(data):
    """Calculate and display subsample sizes for each round and treatment."""
    print("\n=== SUBSAMPLE SIZES (Groups with ≥2 Truthful Reports) ===")

    treatments = ['DA', 'TTC']
    size_data = []

    for treatment in treatments:
        treatment_data = data[data['treatment'] == treatment]

        for round_num in range(1, 21):
            subsample_col = f'subsample_2truth_r{round_num}'
            if subsample_col in treatment_data.columns:
                subsample_size = treatment_data[subsample_col].sum()
                total_size = len(treatment_data)
                percentage = (subsample_size / total_size * 100) if total_size > 0 else 0

                size_data.append({
                    'treatment': treatment,
                    'round': round_num,
                    'subsample_size': subsample_size,
                    'total_size': total_size,
                    'percentage': percentage,
                    'round_group': "Rounds 1-10" if round_num <= 10 else "Rounds 11-20"
                })

    size_df = pd.DataFrame(size_data)

    # Display summary by round groups
    for round_group in ["Rounds 1-10", "Rounds 11-20"]:
        print(f"\n{round_group}:")
        group_data = size_df[size_df['round_group'] == round_group]
        for treatment in treatments:
            treatment_group = group_data[group_data['treatment'] == treatment]
            if not treatment_group.empty:
                avg_subsample = treatment_group['subsample_size'].mean()
                avg_total = treatment_group['total_size'].mean()
                avg_percentage = treatment_group['percentage'].mean()
                print(
                    f"{treatment}: Average subsample size = {avg_subsample:.1f} out of {avg_total:.1f} ({avg_percentage:.1f}%)")

    return size_df


########################################################################################################################
# MAIN ANALYSIS
########################################################################################################################

print("=== STABILITY ANALYSIS WITH SUBSAMPLE ===")

# Generate all required variables
data = generate_truthful_report_variables(data)
data = generate_subsample_2truth_variables(data)
data = generate_blocking_pair_variables(data)
data = generate_envy_free_variables(data)
data = generate_stability_variables(data)

# Display subsample sizes
subsample_sizes = calculate_subsample_sizes(data)

print("\n" + "=" * 80)
print("ORIGINAL ANALYSIS (ALL SUBJECTS)")
print("=" * 80)

# 1. Blocking pairs analysis (Original)
print("\n1. ENVY-FREE STUDENTS- ALL SUBJECTS")

# 1.1: All subjects
rates_envy_free_all = calculate_rates_by_treatment(data, 'envy_free', "all")

# Calculate sample sizes
da_all_count = len(data[data['treatment'] == 'DA'])
ttc_all_count = len(data[data['treatment'] == 'TTC'])

plot_rates(rates_envy_free_all,
           'Proportion of Envy-Free Students by Treatment: All Subjects',
           'envy_free_proportion_all_subjects.png',
           'Proportion of Envy-Free Students',
           {'rounds_1_10': f'n_TTC:{ttc_all_count}; n_DA:{da_all_count}',
            'rounds_11_20': f'n_TTC:{ttc_all_count}; n_DA:{da_all_count}'})

analyze_stability_measure(data, 'envy_free', 'Envy-Free Students', 'all')

# 1.2: Incomplete preferences
rates_envy_free_incomplete = calculate_rates_by_treatment(data, 'envy_free', "incomplete")

# Calculate sample sizes for incomplete preferences
da_inc_1_10 = len(data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 1)])
ttc_inc_1_10 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 1)])
da_inc_11_20 = len(data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 1)])
ttc_inc_11_20 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 1)])

plot_rates(rates_envy_free_incomplete,
           'Proportion of Envy-Free Students by Treatment: Incomplete Preferences',
           'envy_free_proportion_incomplete_preferences.png',
           'Proportion of Envy-Free Students',
           {'rounds_1_10': f'n_TTC:{ttc_inc_1_10}; n_DA:{da_inc_1_10}',
            'rounds_11_20': f'n_TTC:{ttc_inc_11_20}; n_DA:{da_inc_11_20}'})

analyze_stability_measure(data, 'envy_free', 'Envy-Free Students', 'incomplete')

# 1.3: Complete preferences
rates_envy_free_complete = calculate_rates_by_treatment(data, 'envy_free', "complete")

# Calculate sample sizes for complete preferences
da_comp_1_10 = len(data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 0)])
ttc_comp_1_10 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 0)])
da_comp_11_20 = len(data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 0)])
ttc_comp_11_20 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 0)])

plot_rates(rates_envy_free_complete,
           'Proportion of Envy-Free Students by Treatment: Complete Preferences',
           'envy_free_proportion_complete_preferences.png',
           'Proportion of Envy-Free Students',
           {'rounds_1_10': f'n_TTC:{ttc_comp_1_10}; n_DA:{da_comp_1_10}',
            'rounds_11_20': f'n_TTC:{ttc_comp_11_20}; n_DA:{da_comp_11_20}'})

analyze_stability_measure(data, 'envy_free', 'Envy-Free Students', 'complete')


# 1.4: Other preferences
rates_envy_free_other = calculate_rates_by_treatment(data, 'envy_free', "other")

# Calculate sample sizes for other preferences
da_other_1_10 = len(data[(data['treatment'] == 'DA') & (data['incomplete_1'] == 2)])
ttc_other_1_10 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_1'] == 2)])
da_other_11_20 = len(data[(data['treatment'] == 'DA') & (data['incomplete_2'] == 2)])
ttc_other_11_20 = len(data[(data['treatment'] == 'TTC') & (data['incomplete_2'] == 2)])

plot_rates(rates_envy_free_other,
           'Proportion of Envy-Free Students by Treatment: Other Preferences',
           'envy_free_proportion_other_preferences.png',
           'Proportion of Envy-Free Students',
           {'rounds_1_10': f'n_TTC:{ttc_other_1_10}; n_DA:{da_other_1_10}',
            'rounds_11_20': f'n_TTC:{ttc_other_11_20}; n_DA:{da_other_11_20}'})

analyze_stability_measure(data, 'envy_free', 'Envy-Free Students', 'other')

# 2. Group stability analysis (Original)
print("\n2. GROUP STABILITY ANALYSIS - ALL SUBJECTS")

# 2.1: All subjects
rates_stability_all = calculate_group_stability_rates(data)

plot_rates(rates_stability_all,
           'Proportion of Stable Groups by Treatment: All Subjects',
           'stable_groups_proportion_all_subjects.png',
           'Proportion of Stable Groups',
           {'rounds_1_10': f'n_groups_TTC:{ttc_all_count // 3}; n_groups_DA:{da_all_count // 3}',
            'rounds_11_20': f'n_groups_TTC:{ttc_all_count // 3}; n_groups_DA:{da_all_count // 3}'})

analyze_stability_measure(data, 'stable_group', 'Group Stability', 'all', is_group_level=True)

print("\n" + "=" * 80)
print("SUBSAMPLE ANALYSIS (GROUPS WITH ≥2 TRUTHFUL REPORTS)")
print("=" * 80)

# 3. Blocking pairs analysis (Subsample)
print("\n3. BLOCKING PAIRS ANALYSIS - SUBSAMPLE")

# 3.1: All subjects (subsample)
rates_envy_free_all_sub = calculate_rates_by_treatment(data, 'envy_free', "all", subsample_filter=True)

# Calculate subsample sizes for plotting
da_sub_sizes = \
subsample_sizes[(subsample_sizes['treatment'] == 'DA') & (subsample_sizes['round_group'] == 'Rounds 11-20')][
    'subsample_size'].mean()
ttc_sub_sizes = \
subsample_sizes[(subsample_sizes['treatment'] == 'TTC') & (subsample_sizes['round_group'] == 'Rounds 11-20')][
    'subsample_size'].mean()

plot_rates(rates_envy_free_all_sub,
           'Proportion of Envy-Free Students by Treatment: All Subjects (Subsample: ≥2 Truthful)',
           'envy_free_proportion_all_subjects_subsample.png',
           'Proportion of Envy-Free Students',
           {'rounds_1_10': 'Subsample sizes vary by round',
            'rounds_11_20': 'Subsample sizes vary by round'})

analyze_stability_measure(data, 'envy_free', 'Envy-Free Students', 'all', subsample_filter=True)

# 3.2: Incomplete preferences (subsample)
rates_envy_free_incomplete_sub = calculate_rates_by_treatment(data, 'envy_free', "incomplete", subsample_filter=True)

plot_rates(rates_envy_free_incomplete_sub,
           'Proportion of Envy-Free Students by Treatment: Incomplete Preferences (Subsample: ≥2 Truthful)',
           'envy_free_proportion_incomplete_preferences_subsample.png',
           'Proportion of Envy-Free Students',
           {'rounds_1_10': 'Subsample sizes vary by round',
            'rounds_11_20': 'Subsample sizes vary by round'})

analyze_stability_measure(data, 'envy_free', 'Envy-Free Students', 'incomplete', subsample_filter=True)

# 3.3: Complete preferences (subsample)
rates_envy_free_complete_sub = calculate_rates_by_treatment(data, 'envy_free', "complete", subsample_filter=True)

plot_rates(rates_envy_free_complete_sub,
           'Proportion of Envy-Free Students by Treatment: Complete Preferences (Subsample: ≥2 Truthful)',
           'envy_free_proportion_complete_preferences_subsample.png',
           'Proportion of Envy-Free Students',
           {'rounds_1_10': 'Subsample sizes vary by round',
            'rounds_11_20': 'Subsample sizes vary by round'})

analyze_stability_measure(data, 'envy_free', 'Envy-Free Students', 'complete', subsample_filter=True)

# 3.4: Other preferences (subsample)
rates_envy_free_other_sub = calculate_rates_by_treatment(data, 'envy_free', "other", subsample_filter=True)

plot_rates(rates_envy_free_other_sub,
           'Proportion of Envy-Free Students by Treatment: Other Preferences (Subsample: ≥2 Truthful)',
           'envy_free_proportion_other_preferences_subsample.png',
           'Proportion of Envy-Free Students',
           {'rounds_1_10': 'Subsample sizes vary by round',
            'rounds_11_20': 'Subsample sizes vary by round'})

analyze_stability_measure(data, 'envy_free', 'Envy-Free Students', 'other', subsample_filter=True)

# 4. Group stability analysis (Subsample)
print("\n4. GROUP STABILITY ANALYSIS - SUBSAMPLE")

# 4.1: All subjects (subsample)
rates_stability_all_sub = calculate_group_stability_rates(data, subsample_filter=True)

plot_rates(rates_stability_all_sub,
           'Proportion of Stable Groups by Treatment: All Subjects (Subsample: ≥2 Truthful)',
           'stable_groups_proportion_all_subjects_subsample.png',
           'Proportion of Stable Groups',
           {'rounds_1_10': 'Group sizes vary by round',
            'rounds_11_20': 'Group sizes vary by round'})

analyze_stability_measure(data, 'stable_group', 'Group Stability', 'all', is_group_level=True, subsample_filter=True)

# Save the data
print("\nAnalysis complete. Data saved to 'combined_with_stability_analysis.csv'")
data.to_csv('combined_with_stability_analysis.csv', index=False)