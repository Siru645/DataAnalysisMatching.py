import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
data = pd.read_csv(
    r'C:\Users\lsr64\OneDrive - Georgia State University\Georgia State University\My Research\Projects\Matching with incomplete preferences\Data analysis\combined.csv')


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

data = generate_truthful_report_variables(data)

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
        data[f'weak_true_report_r{round_num}'] = (xy_correct & xz_correct & yz_correct).astype(int)

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
        data[f'weak_true_report_r{round_num}'] = (xy_correct & xz_correct & yz_correct).astype(int)

    return data

data = generate_weak_truthful_report_variables(data)

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

data = generate_subsample_2truth_variables(data)

def is_indifferent(pref_lists):
    """Check if a student has indifferent preferences."""
    return pref_lists == "Indifference"


def get_non_dominated_schools(pref_lists):
    """Get the non-dominated schools from preference lists."""
    if is_indifferent(pref_lists):
        return {'X', 'Y', 'Z'}  # All schools for indifferent students

    rankings = pref_lists.split(';')
    first_choices = set()

    for ranking in rankings:
        choices = ranking.split(',')
        if len(choices) >= 1:
            first_choices.add(choices[0])

    return first_choices


def prefers_other_assignment(own_pref_lists, own_assignment, other_assignment):
    """Check if a student prefers another student's assignment over their own in ALL preference rankings."""
    if is_indifferent(own_pref_lists):
        return False

    rankings = own_pref_lists.split(';')

    for ranking in rankings:
        choices = ranking.split(',')
        first_choice, second_choice, third_choice = choices[0], choices[1], choices[2]

        # Check if student prefers the other's assignment in this ranking
        prefers_in_this_ranking = (
                (first_choice == other_assignment and second_choice == own_assignment) or
                (first_choice == other_assignment and third_choice == own_assignment) or
                (second_choice == other_assignment and third_choice == own_assignment)
        )

        if not prefers_in_this_ranking:
            return False

    return True


def identify_students_in_swaps(group_df, round_num):
    """Returns a list of students who can strictly benefit from swapping."""
    group_df = group_df.reset_index(drop=True)

    # Get the correct preference lists and assignments
    pref_col = 'preference_1_lists' if round_num <= 10 else 'preference_2_lists'
    assignment_col = f'assigned_school_r{round_num}'

    if assignment_col not in group_df.columns or pref_col not in group_df.columns:
        return []

    assignments = group_df[assignment_col].tolist()
    pref_lists = group_df[pref_col].tolist()

    # Count indifferent students
    indifferent_mask = [is_indifferent(pref) for pref in pref_lists]
    num_indifferent = sum(indifferent_mask)

    students_in_swaps = set()

    if num_indifferent == 3:
        return []  # No beneficial swaps possible
    elif num_indifferent == 2:
        # Find the non-indifferent student
        non_indiff_idx = indifferent_mask.index(False)
        non_indiff_pref = pref_lists[non_indiff_idx]
        non_indiff_assignment = assignments[non_indiff_idx]

        # Check if the non-indifferent student got any non-dominated school
        non_dominated_schools = get_non_dominated_schools(non_indiff_pref)
        if non_indiff_assignment not in non_dominated_schools:
            students_in_swaps.add(non_indiff_idx)
    elif num_indifferent == 1:
        # Get the two non-indifferent students
        non_indiff_indices = [i for i, is_indiff in enumerate(indifferent_mask) if not is_indiff]

        if len(non_indiff_indices) == 2:
            idx1, idx2 = non_indiff_indices

            # Check for mutual preference for swap
            student1_prefers_2 = prefers_other_assignment(pref_lists[idx1], assignments[idx1], assignments[idx2])
            student2_prefers_1 = prefers_other_assignment(pref_lists[idx2], assignments[idx2], assignments[idx1])

            if student1_prefers_2 and student2_prefers_1:
                students_in_swaps.update([idx1, idx2])
    else:
        # No students are indifferent - check for cycles and swaps
        # Check for three-way cycles
        s1_prefers_s2 = prefers_other_assignment(pref_lists[0], assignments[0], assignments[1])
        s2_prefers_s3 = prefers_other_assignment(pref_lists[1], assignments[1], assignments[2])
        s3_prefers_s1 = prefers_other_assignment(pref_lists[2], assignments[2], assignments[0])

        s1_prefers_s3 = prefers_other_assignment(pref_lists[0], assignments[0], assignments[2])
        s3_prefers_s2 = prefers_other_assignment(pref_lists[2], assignments[2], assignments[1])
        s2_prefers_s1 = prefers_other_assignment(pref_lists[1], assignments[1], assignments[0])

        # Check for three-way cycles
        if (s1_prefers_s2 and s2_prefers_s3 and s3_prefers_s1) or \
                (s1_prefers_s3 and s3_prefers_s2 and s2_prefers_s1):
            students_in_swaps.update([0, 1, 2])
        else:
            # Check for two-way swaps
            pairs = [(0, 1), (0, 2), (1, 2)]
            for i, j in pairs:
                i_prefers_j = prefers_other_assignment(pref_lists[i], assignments[i], assignments[j])
                j_prefers_i = prefers_other_assignment(pref_lists[j], assignments[j], assignments[i])

                if i_prefers_j and j_prefers_i:
                    students_in_swaps.update([i, j])

    return list(students_in_swaps)


def generate_non_dominated_variables(data):
    """Generate non_dominated_r1 to non_dominated_r20 variables."""
    print("Generating non-dominated assignment variables...")

    # Prepare all non-dominated columns at once
    non_dominated_columns = {}

    for round_num in range(1, 21):
        assignment_col = f'assigned_school_r{round_num}'
        pref_col = 'preference_1_lists' if round_num <= 10 else 'preference_2_lists'

        if assignment_col not in data.columns or pref_col not in data.columns:
            non_dominated_columns[f'non_dominated_r{round_num}'] = 0
            continue

        # Calculate non-dominated assignments
        non_dominated = []
        for idx, student in data.iterrows():
            student_pref = student[pref_col]
            student_assignment = student[assignment_col]

            if pd.isna(student_assignment) or pd.isna(student_pref):
                non_dominated.append(0)
                continue

            non_dominated_schools = get_non_dominated_schools(student_pref)
            non_dominated.append(1 if student_assignment in non_dominated_schools else 0)

        non_dominated_columns[f'non_dominated_r{round_num}'] = non_dominated

    # Add all non-dominated columns at once
    non_dominated_df = pd.DataFrame(non_dominated_columns, index=data.index)
    data = pd.concat([data, non_dominated_df], axis=1)

    return data

data = generate_non_dominated_variables(data)

def generate_swap_variables(data):
    """Generate swap_r1 to swap_r20 variables."""
    print("Generating swap variables...")

    # Initialize all swap columns at once
    swap_columns = {}
    for round_num in range(1, 21):
        swap_columns[f'swap_r{round_num}'] = 0

    # Add all columns at once
    swap_df = pd.DataFrame(swap_columns, index=data.index)
    data = pd.concat([data, swap_df], axis=1)

    for round_num in range(1, 21):
        group_col = f'group_id_r{round_num}'

        if group_col not in data.columns:
            continue

        for group_id, group_data in data.groupby(group_col):
            if pd.isna(group_id) or len(group_data) != 3:
                continue

            students_in_swaps = identify_students_in_swaps(group_data, round_num)

            if students_in_swaps:
                group_indices = group_data.index.tolist()
                for student_idx in students_in_swaps:
                    if student_idx < len(group_indices):
                        actual_index = group_indices[student_idx]
                        data.loc[actual_index, f'swap_r{round_num}'] = 1

    return data

data = generate_swap_variables(data)

def generate_efficiency_variables(data):
    """Generate count_swap and efficient_group variables."""
    print("Generating efficiency variables...")

    # Prepare all new columns at once
    count_swap_columns = {}
    efficient_columns = {}

    # Generate count_swap variables
    for round_num in range(1, 21):
        group_col = f'group_id_r{round_num}'
        swap_col = f'swap_r{round_num}'
        count_swap_col = f'count_swap_r{round_num}'

        if group_col not in data.columns or swap_col not in data.columns:
            count_swap_columns[count_swap_col] = 0
        else:
            # Calculate count of students with swap=1 for each group
            group_swap_counts = data.groupby(group_col)[swap_col].sum()
            count_swap_columns[count_swap_col] = data[group_col].map(group_swap_counts).fillna(0)

    # Add count_swap columns
    count_swap_df = pd.DataFrame(count_swap_columns, index=data.index)
    data = pd.concat([data, count_swap_df], axis=1)

    # Generate efficient_group variables
    for round_num in range(1, 21):
        count_swap_col = f'count_swap_r{round_num}'
        efficient_col = f'efficient_group_r{round_num}'

        if count_swap_col in data.columns:
            efficient_columns[efficient_col] = (data[count_swap_col] == 0).astype(int)
        else:
            efficient_columns[efficient_col] = 0

    # Add efficient_group columns
    efficient_df = pd.DataFrame(efficient_columns, index=data.index)
    data = pd.concat([data, efficient_df], axis=1)

    return data

data = generate_efficiency_variables(data)

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

data = generate_blocking_pair_variables(data)

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

data = generate_envy_free_variables(data)


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

data = generate_stability_variables(data)

data.to_csv(r'C:\Users\lsr64\OneDrive - Georgia State University\Georgia State University\My Research\Projects\Matching with incomplete preferences\Data analysis\updated.csv', index=False)