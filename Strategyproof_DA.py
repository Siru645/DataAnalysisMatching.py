import copy
import random
import pandas as pd
from collections import Counter, defaultdict

# Define the preference lists for each student (from Test1.xlsx)
student_preferences = {
    'a': [
        'X,Y,Z',
        'X,Z,Y',
        'Z,X,Y',
        'Z,X,Y',
        'Y,X,Z',
        'Y,Z,X',
        'X,Y,Z;X,Z,Y',
        'X,Z,Y;Z,X,Y',
        'Y,Z,X;Z,Y,X',
        'Y,X,Z;Y,Z,X',
        'X,Y,Z;Y,X,Z',
        'Z,X,Y;Z,Y,X',
        'X,Z,Y;Z,X,Y;Z,Y,X',
        'X,Y,Z;Y,X,Z;Y,Z,X',
        'Z,X,Y;Z,Y,X;Y,Z,X',
        'X,Y,Z;X,Z,Y;Y,X,Z',
        'Y,X,Z;Y,Z,X;Z,Y,X',
        'X,Y,Z;X,Z,Y;Z,X,Y',
        'X,Y,Z;X,Z,Y;Y,X,Z;Y,Z,X;Z,X,Y;Z,Y,X'
    ],
    'b': [
        'X,Y,Z',
        'X,Z,Y',
        'Z,X,Y',
        'Z,X,Y',
        'Y,X,Z',
        'Y,Z,X',
        'X,Y,Z;X,Z,Y',
        'X,Z,Y;Z,X,Y',
        'Y,Z,X;Z,Y,X',
        'Y,X,Z;Y,Z,X',
        'X,Y,Z;Y,X,Z',
        'Z,X,Y;Z,Y,X',
        'X,Z,Y;Z,X,Y;Z,Y,X',
        'X,Y,Z;Y,X,Z;Y,Z,X',
        'Z,X,Y;Z,Y,X;Y,Z,X',
        'X,Y,Z;X,Z,Y;Y,X,Z',
        'Y,X,Z;Y,Z,X;Z,Y,X',
        'X,Y,Z;X,Z,Y;Z,X,Y',
        'X,Y,Z;X,Z,Y;Y,X,Z;Y,Z,X;Z,X,Y;Z,Y,X'
    ],
    'c': [
        'X,Y,Z',
        'X,Z,Y',
        'Z,X,Y',
        'Z,X,Y',
        'Y,X,Z',
        'Y,Z,X',
        'X,Y,Z;X,Z,Y',
        'X,Z,Y;Z,X,Y',
        'Y,Z,X;Z,Y,X',
        'Y,X,Z;Y,Z,X',
        'X,Y,Z;Y,X,Z',
        'Z,X,Y;Z,Y,X',
        'X,Z,Y;Z,X,Y;Z,Y,X',
        'X,Y,Z;Y,X,Z;Y,Z,X',
        'Z,X,Y;Z,Y,X;Y,Z,X',
        'X,Y,Z;X,Z,Y;Y,X,Z',
        'Y,X,Z;Y,Z,X;Z,Y,X',
        'X,Y,Z;X,Z,Y;Z,X,Y',
        'X,Y,Z;X,Z,Y;Y,X,Z;Y,Z,X;Z,X,Y;Z,Y,X'
    ]
}

# School preferences (fixed)
SchPrefs = {
    'X': ['a', 'c', 'b'],
    'Y': ['b', 'a', 'c'],
    'Z': ['c', 'b', 'a']
}

# Preference sets from Test3.xlsx (hardcoded)
preference_sets = {
    1: ({'X'}, {'Y'}, {'Z'}),
    2: ({'X'}, {'Z'}, {'Y'}),
    3: ({'Z'}, {'X'}, {'Y'}),
    4: ({'Z'}, {'X'}, {'Y'}),
    5: ({'Y'}, {'X'}, {'Z'}),
    6: ({'Y'}, {'Z'}, {'X'}),
    7: ({'X'}, {'Y', 'Z'}, set()),
    8: ({'X', 'Z'}, {'Y'}, set()),
    9: ({'Y', 'Z'}, {'X'}, set()),
    10: ({'Y'}, {'X', 'Z'}, set()),
    11: ({'X', 'Y'}, {'Z'}, set()),
    12: ({'Z'}, {'X', 'Y'}, set()),
    13: ({'Z'}, {'Y'}, set()),
    14: ({'Y'}, {'Z'}, set()),
    15: ({'Z'}, {'X'}, set()),
    16: ({'X'}, {'Z'}, set()),
    17: ({'Y'}, {'X'}, set()),
    18: ({'X'}, {'Y'}, set()),
    19: ({'X', 'Y', 'Z'}, set(), set())
}


def DA(StPrefs, SchPrefs):
    """Deferred Acceptance algorithm for 3 students and 3 schools"""
    Cpcty = {'X': 1, 'Y': 1, 'Z': 1}
    Students = list(StPrefs.keys())

    for i in StPrefs.keys():
        if i not in StPrefs[i]:
            StPrefs[i].append(i)

    App = {}
    for i in Students:
        for j in SchPrefs.keys():
            App[(j, i)] = 0

    NumRej = {}
    for i in Students:
        NumRej[i] = 0

    Seeking = set(Students)
    Receiving = set()
    ResStud = {}

    while Seeking:
        for i in Seeking:
            j = StPrefs[i][NumRej[i]]
            App[(j, i)] = 1
            Receiving.add(j)

        for j in Receiving:
            Applicants = [i for i in Students if App[(j, i)] == 1]

            if len(Applicants) > Cpcty[j]:
                sorted_applicants = []
                for p in SchPrefs[j][::-1]:
                    if p in Applicants:
                        sorted_applicants.append(p)

                to_reject = sorted_applicants[:len(Applicants) - Cpcty[j]]
                for i in to_reject:
                    App[(j, i)] = 0
                    NumRej[i] += 1
                    Seeking.add(i)

            for i in Applicants:
                if i in Seeking and App[(j, i)] == 1:
                    Seeking.remove(i)

        Receiving = set()

    for i in Students:
        if i not in ResStud:
            assigned_school = StPrefs[i][NumRej[i]]
            ResStud[i] = assigned_school

    return ResStud


def prefers_school_over_another(preference_list, school1, school2):
    """Check if a student prefers school1 over school2."""
    try:
        rank1 = preference_list.index(school1)
        rank2 = preference_list.index(school2)
        return rank1 < rank2
    except ValueError:
        return False


def check_two_way_swap(students, matching, parsed_pref_lists):
    """Check if any two students would prefer to swap schools in ALL their preference lists."""
    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            student1 = students[i]
            student2 = students[j]

            both_prefer_swap = (
                    all(prefers_school_over_another(pref_list, matching[student2], matching[student1])
                        for pref_list in parsed_pref_lists[student1]) and
                    all(prefers_school_over_another(pref_list, matching[student1], matching[student2])
                        for pref_list in parsed_pref_lists[student2])
            )

            if both_prefer_swap:
                return True

    return False


def check_three_way_swap(students, matching, parsed_pref_lists):
    """Check if all three students would prefer a 3-way swap in ALL their preference lists."""
    forward_cycle = (
            all(prefers_school_over_another(pref_list, matching[students[1]], matching[students[0]])
                for pref_list in parsed_pref_lists[students[0]]) and
            all(prefers_school_over_another(pref_list, matching[students[2]], matching[students[1]])
                for pref_list in parsed_pref_lists[students[1]]) and
            all(prefers_school_over_another(pref_list, matching[students[0]], matching[students[2]])
                for pref_list in parsed_pref_lists[students[2]])
    )

    backward_cycle = (
            all(prefers_school_over_another(pref_list, matching[students[2]], matching[students[0]])
                for pref_list in parsed_pref_lists[students[0]]) and
            all(prefers_school_over_another(pref_list, matching[students[0]], matching[students[1]])
                for pref_list in parsed_pref_lists[students[1]]) and
            all(prefers_school_over_another(pref_list, matching[students[1]], matching[students[2]])
                for pref_list in parsed_pref_lists[students[2]])
    )

    return forward_cycle or backward_cycle


def count_students_willing_to_swap(matching, student_pref_lists):
    """Count how many students would like to swap their allocations."""
    students = list(matching.keys())
    if len(students) != 3:
        return 0

    parsed_pref_lists = {}
    for student_role, pref_lists in student_pref_lists.items():
        parsed_pref_lists[student_role] = []
        for pref_str in pref_lists:
            parsed_list = [school.strip() for school in pref_str.split(',')]
            parsed_pref_lists[student_role].append(parsed_list)

    if check_three_way_swap(students, matching, parsed_pref_lists):
        return 3

    if check_two_way_swap(students, matching, parsed_pref_lists):
        return 2

    return 0


def get_all_matchings_with_swap_counts(student_pref_lists, mechanism=DA):
    """
    Get ALL possible matchings with their swap counts, not just the best ones.
    Returns a list of all matchings with their frequencies.
    """

    def generate_all_combinations(roles, student_lists, current_combo=None, index=0):
        if current_combo is None:
            current_combo = {}

        if index == len(roles):
            return [current_combo.copy()]

        current_role = roles[index]
        combinations = []

        for pref_list in student_lists[current_role]:
            current_combo[current_role] = pref_list
            combinations.extend(generate_all_combinations(roles, student_lists, current_combo, index + 1))

        return combinations

    student_roles = list(student_pref_lists.keys())
    all_combinations = generate_all_combinations(student_roles, student_pref_lists)

    # Run the matching mechanism for each combination
    all_matchings = []
    for combo in all_combinations:
        # Format for the matching mechanism
        StPrefs = {}
        for role, preferences_str in combo.items():
            preferences_list = [p.strip() for p in preferences_str.split(',')]
            StPrefs[role] = preferences_list

        # Create deep copies
        StPrefs_copy = copy.deepcopy(StPrefs)
        SchPrefs_copy = copy.deepcopy(SchPrefs)

        # Run the mechanism
        result = mechanism(StPrefs_copy, SchPrefs_copy)
        swap_count = count_students_willing_to_swap(result, student_pref_lists)

        all_matchings.append({
            'preferences': combo,
            'matching': result,
            'swap_count': swap_count
        })

    return all_matchings


def calculate_assignment_probabilities(all_matchings, student='a'):
    """
    Calculate probability distribution over school assignments.
    Returns a dictionary {school: probability}.
    """
    # Group matchings by swap count (since we select from minimum swap count)
    swap_groups = defaultdict(list)
    for matching_data in all_matchings:
        swap_count = matching_data['swap_count']
        swap_groups[swap_count].append(matching_data)

    # Find minimum swap count
    min_swap_count = min(swap_groups.keys())
    best_matchings = swap_groups[min_swap_count]

    # Count assignments for the student
    assignment_counts = Counter()
    for matching_data in best_matchings:
        assigned_school = matching_data['matching'].get(student, None)
        if assigned_school:
            assignment_counts[assigned_school] += 1

    # Convert to probabilities
    total_count = sum(assignment_counts.values())
    probabilities = {}
    for school, count in assignment_counts.items():
        probabilities[school] = count / total_count if total_count > 0 else 0

    return probabilities


def calculate_set_probabilities(probabilities, school_sets):
    """
    Calculate probabilities for each preference set.
    """
    best_set, second_set, third_set = school_sets

    prob_best = sum(probabilities.get(school, 0) for school in best_set)
    prob_second = sum(probabilities.get(school, 0) for school in second_set)
    prob_third = sum(probabilities.get(school, 0) for school in third_set)

    return prob_best, prob_second, prob_third


def calculate_ratio_first_to_second(probabilities, school_sets):
    """
    Calculate the ratio of probability of first set to second set.
    Returns 0 if both probabilities are 0 (0/0 case).
    """
    best_set, second_set, third_set = school_sets

    prob_best = sum(probabilities.get(school, 0) for school in best_set)
    prob_second = sum(probabilities.get(school, 0) for school in second_set)

    if prob_best == 0 and prob_second == 0:
        return 0
    elif prob_second == 0:
        return float('inf') if prob_best > 0 else 0
    else:
        return prob_best / prob_second


def is_strategic_dominant_cases_1_12_19(truthful_probs, strategic_probs):
    """
    Check if strategic reporting is STRICTLY dominant over truthful reporting for cases 1-12 and 19.
    Strategic is dominant if:
    1. P(best_set | strategic) > P(best_set | truthful) OR
    2. P(best_set | strategic) = P(best_set | truthful) AND P(best_set ∪ second_set | strategic) > P(best_set ∪ second_set | truthful)

    Returns True if strategic dominates truthful (i.e., truthful is dominated).
    """
    truthful_best, truthful_second, truthful_third = truthful_probs
    strategic_best, strategic_second, strategic_third = strategic_probs

    # Strategic strictly dominates if it gives strictly higher probability for best set
    if strategic_best > truthful_best:
        return True

    # If best set probabilities are equal, check if strategic gives strictly higher probability for best ∪ second
    if strategic_best == truthful_best and (strategic_best + strategic_second) > (truthful_best + truthful_second):
        return True

    return False


def is_strategic_dominant_cases_13_18(truthful_ratio, strategic_ratio):
    """
    Check if strategic reporting is dominant for cases 13-18.
    Strategic is dominant if the ratio of truthful report is smaller than the manipulation report.
    If truthful_ratio >= strategic_ratio, then truthful is non-dominated.

    Returns True if strategic dominates truthful (i.e., truthful is dominated).
    """
    return strategic_ratio > truthful_ratio


def analyze_strategyproofness():
    """
    Analyze whether the mechanism is strategyproof using probability comparison.
    """
    print("=" * 80)
    print("STRATEGYPROOFNESS ANALYSIS WITH PROBABILITY COMPARISON")
    print("=" * 80)

    analysis_results = []

    # For each possible true preference of student 'a'
    for true_a_idx in range(19):
        print(f"\n--- Analyzing when student a's TRUE preference is option {true_a_idx + 1} ---")

        true_a_pref = student_preferences['a'][true_a_idx]

        # Get preference sets from hardcoded data
        school_sets = preference_sets[true_a_idx + 1]
        print(f"  Preference sets - Best: {school_sets[0]}, Second: {school_sets[1]}, Third: {school_sets[2]}")

        dominated_count = 0
        non_dominated_count = 0

        # For each combination of b and c's preferences
        for b_option_idx in range(19):
            for c_option_idx in range(19):
                print(f"  Checking b[{b_option_idx + 1}]c[{c_option_idx + 1}]:", end=" ")

                # Get truthful assignment probabilities
                truthful_combo = {
                    'a': student_preferences['a'][true_a_idx].split(';'),
                    'b': student_preferences['b'][b_option_idx].split(';'),
                    'c': student_preferences['c'][c_option_idx].split(';')
                }

                truthful_matchings = get_all_matchings_with_swap_counts(truthful_combo)
                truthful_probabilities = calculate_assignment_probabilities(truthful_matchings, student='a')
                truthful_set_probs = calculate_set_probabilities(truthful_probabilities, school_sets)

                # Check all possible strategic reports by student a
                is_dominated = False
                dominating_reports = []

                for strategic_a_idx in range(19):
                    if strategic_a_idx == true_a_idx:  # Skip truthful report
                        continue

                    # Get strategic assignment probabilities
                    strategic_combo = {
                        'a': student_preferences['a'][strategic_a_idx].split(';'),
                        'b': student_preferences['b'][b_option_idx].split(';'),
                        'c': student_preferences['c'][c_option_idx].split(';')
                    }

                    strategic_matchings = get_all_matchings_with_swap_counts(strategic_combo)
                    strategic_probabilities = calculate_assignment_probabilities(strategic_matchings, student='a')
                    strategic_set_probs = calculate_set_probabilities(strategic_probabilities, school_sets)

                    # Check if strategic reporting dominates truthful reporting
                    # Use different methods for different cases
                    case_number = true_a_idx + 1

                    if case_number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19]:
                        # Use original method for cases 1-12 and 19
                        if is_strategic_dominant_cases_1_12_19(truthful_set_probs, strategic_set_probs):
                            is_dominated = True
                            dominating_reports.append({
                                'strategic_report': strategic_a_idx + 1,
                                'strategic_probs': strategic_set_probs,
                                'strategic_assignments': strategic_probabilities
                            })
                    elif case_number in [13, 14, 15, 16, 17, 18]:
                        # Use ratio method for cases 13-18
                        truthful_ratio = calculate_ratio_first_to_second(truthful_probabilities, school_sets)
                        strategic_ratio = calculate_ratio_first_to_second(strategic_probabilities, school_sets)

                        if is_strategic_dominant_cases_13_18(truthful_ratio, strategic_ratio):
                            is_dominated = True
                            dominating_reports.append({
                                'strategic_report': strategic_a_idx + 1,
                                'strategic_probs': strategic_set_probs,
                                'strategic_assignments': strategic_probabilities,
                                'truthful_ratio': truthful_ratio,
                                'strategic_ratio': strategic_ratio
                            })

                if is_dominated:
                    dominated_count += 1
                    print(f"DOMINATED")
                    print(
                        f"    Truthful probs: Best={truthful_set_probs[0]:.3f}, Second={truthful_set_probs[1]:.3f}, Third={truthful_set_probs[2]:.3f}")
                    for report in dominating_reports[:1]:  # Show first dominating report
                        probs = report['strategic_probs']
                        print(
                            f"    Strategic {report['strategic_report']} probs: Best={probs[0]:.3f}, Second={probs[1]:.3f}, Third={probs[2]:.3f}")
                        if case_number in [13, 14, 15, 16, 17, 18] and 'truthful_ratio' in report:
                            print(
                                f"    Truthful ratio: {report['truthful_ratio']:.3f}, Strategic ratio: {report['strategic_ratio']:.3f}")
                else:
                    non_dominated_count += 1
                    print(f"Non-dominated")
                    print(
                        f"    Truthful probs: Best={truthful_set_probs[0]:.3f}, Second={truthful_set_probs[1]:.3f}, Third={truthful_set_probs[2]:.3f}")

                # Store detailed result
                analysis_results.append({
                    'true_a_option': true_a_idx + 1,
                    'b_option': b_option_idx + 1,
                    'c_option': c_option_idx + 1,
                    'combination': f"a[{true_a_idx + 1}]b[{b_option_idx + 1}]c[{c_option_idx + 1}]",
                    'is_dominated': is_dominated,
                    'truthful_prob_best': truthful_set_probs[0],
                    'truthful_prob_second': truthful_set_probs[1],
                    'truthful_prob_third': truthful_set_probs[2],
                    'truthful_assignments': str(truthful_probabilities),
                    'dominating_reports_count': len(dominating_reports)
                })

        # Summary for this true preference
        total_scenarios = non_dominated_count + dominated_count
        dominated_percentage = (dominated_count / total_scenarios) * 100 if total_scenarios > 0 else 0

        print(f"\n  Summary for a's true preference {true_a_idx + 1}:")
        print(f"    Non-dominated: {non_dominated_count}/{total_scenarios} ({100 - dominated_percentage:.2f}%)")
        print(f"    Dominated: {dominated_count}/{total_scenarios} ({dominated_percentage:.2f}%)")

    # Convert to DataFrame and save
    df = pd.DataFrame(analysis_results)
    df.to_csv('strategyproof_analysis_probability_modified.csv', index=False)

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    overall_dominated = df['is_dominated'].sum()
    total_scenarios = len(df)
    overall_percentage = (overall_dominated / total_scenarios) * 100

    print(f"Total scenarios analyzed: {total_scenarios}")
    print(f"Scenarios where truthful reporting is dominated: {overall_dominated} ({overall_percentage:.2f}%)")
    print(
        f"Scenarios where truthful reporting is non-dominated: {total_scenarios - overall_dominated} ({100 - overall_percentage:.2f}%)")

    print(f"\nDetailed results saved to 'strategyproof_analysis_probability_modified.csv'")

    return analysis_results


# Run the analysis
if __name__ == "__main__":
    analyze_strategyproofness()