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


def ChainGrow(Chain, AfterChain, StPrefs, SchPrefs):
    """
    Grows the chain by adding the next student and school.
    Returns the next student and a flag indicating if a cycle is found.
    i: student, j:school
    """
    Chain.append(AfterChain)  # Add the current student to the chain
    j = StPrefs[AfterChain][0]  # Get the top choice school for the current student
    Chain.append(j)  # Add the school to the chain
    i = SchPrefs[j][0]  # Get the top choice student for the current school
    Flag = 0
    if i in Chain:  # Check if a cycle is formed
        Flag = 1
    return (i, Flag)


def FindCycle(i, StPrefs, SchPrefs):
    """
    Finds a cycle starting from student i.
    Returns the cycle as a list alternating between students and schools.
    """
    Chain = []
    AfterChain = i
    Flag = 0
    while Flag == 0:
        [i, Flag] = ChainGrow(Chain, AfterChain, StPrefs, SchPrefs)
        AfterChain = i
    Ind = Chain.index(i)  # Find the start of the cycle
    Chain = Chain[Ind:]  # Trim the chain to the cycle
    return Chain


def ProcessCycle(Chain, StPrefs, SchPrefs, ResStud):
    """
    Processes the cycle by assigning students to schools.
    Removes matched students and schools from further consideration.
    """
    JASt = []  # Just Assigned Students
    JASch = []  # Just Assigned Schools
    for l in range(0, len(Chain), 2):  # Iterate over students in the cycle
        i = Chain[l]  # Student
        j = Chain[l + 1]  # School
        ResStud[i] = j  # Assign the student to the school
        JASt.append(i)  # Add the student to the list of assigned students
        JASch.append(j)  # Add the school to the list of assigned schools
    return (JASt, JASch, ResStud)


def Update(JASt, JASch, StPrefs, SchPrefs, ResStud):
    """
    Updates the preference lists by removing assigned students and schools.
    """
    for i in JASt:  # Remove assigned students from StPrefs
        if i in StPrefs:
            del StPrefs[i]

    for j in JASch:  # Remove assigned schools from SchPrefs
        if j in SchPrefs:
            del SchPrefs[j]

    for i in StPrefs.keys():  # Remove unavailable schools from student preferences
        StPrefs[i] = [j for j in StPrefs[i] if j in SchPrefs.keys()]

    for j in SchPrefs.keys():  # Remove unavailable students from school preferences
        SchPrefs[j] = [i for i in SchPrefs[j] if i in StPrefs.keys()]

    return (StPrefs, SchPrefs, ResStud)


def TTC(StPrefs, SchPrefs):
    # Create deep copies of input dictionaries
    StPrefs_copy = copy.deepcopy(StPrefs)
    SchPrefs_copy = copy.deepcopy(SchPrefs)

    stop = 0
    ResStud = {}  # Dictionary to store final assignments
    while stop == 0:
        if not StPrefs_copy:  # If no students are left, stop
            stop = 1
            break
        i = list(StPrefs_copy.keys())[0]  # Start with the first student
        Chain = FindCycle(i, StPrefs_copy, SchPrefs_copy)  # Find a cycle
        [JASt, JASch, ResStud] = ProcessCycle(Chain, StPrefs_copy, SchPrefs_copy, ResStud)  # Process the cycle
        [StPrefs_copy, SchPrefs_copy, ResStud] = Update(JASt, JASch, StPrefs_copy, SchPrefs_copy,
                                                        ResStud)  # Update preferences

    # Remove debug print statement
    # print(ResStud)
    return ResStud


def prefers_school_over_another(preference_list, school1, school2):
    """Check if a student prefers school1 over school2."""
    try:
        rank1 = preference_list.index(school1)
        rank2 = preference_list.index(school2)
        return rank1 < rank2
    except ValueError:
        return False


def count_blocking_pairs(matching, student_pref_lists):
    """
    Count how many students and schools are involved in blocking pairs.
    A blocking pair is a student-school pair where:
    1. The student prefers the school over their current match in ALL their preference lists
    2. The school prefers the student over its currently matched student

    Args:
        matching: Dictionary mapping students to their assigned schools
        student_pref_lists: Dictionary mapping students to all their preference lists

    Returns:
        Tuple of (set of students involved in blocking pairs, set of schools involved in blocking pairs)
    """
    # Parse all preference lists for each student
    parsed_pref_lists = {}
    for student_role, pref_lists in student_pref_lists.items():
        parsed_pref_lists[student_role] = []
        for pref_str in pref_lists:
            # Split and strip each preference list
            parsed_list = [school.strip() for school in pref_str.split(',')]
            parsed_pref_lists[student_role].append(parsed_list)

    # Create inverse matching (school -> student)
    school_matching = {matching[student]: student for student in matching}

    # Find all blocking pairs and collect involved students and schools
    blocking_students = set()
    blocking_schools = set()

    for student in matching:
        current_school = matching[student]

        # Check each school that's not the student's current school
        for school in SchPrefs:
            if school == current_school:
                continue

            # Check if student prefers this school in ALL their preference lists
            prefers_school = all(
                prefers_school_over_another(pref_list, school, current_school)
                for pref_list in parsed_pref_lists[student]
            )

            if prefers_school:
                # Check if school prefers this student over its current match
                current_match = school_matching.get(school)
                if current_match:
                    school_prefs = SchPrefs[school]
                    # Lower index = higher preference
                    if school_prefs.index(student) < school_prefs.index(current_match):
                        blocking_students.add(student)
                        blocking_schools.add(school)
                else:
                    # If school is unmatched, it would accept any student
                    blocking_students.add(student)
                    blocking_schools.add(school)

    return blocking_students, blocking_schools


def get_all_matchings_with_blocking_counts(student_pref_lists, mechanism=TTC):
    """
    Get ALL possible matchings with their blocking pair counts.
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

        # Count blocking pairs
        blocking_students, blocking_schools = count_blocking_pairs(result, student_pref_lists)
        blocking_count = len(blocking_students)

        all_matchings.append({
            'preferences': combo,
            'matching': result,
            'blocking_count': blocking_count
        })

    return all_matchings


def calculate_assignment_probabilities(all_matchings, student='a'):
    """
    Calculate probability distribution over school assignments.
    Returns a dictionary {school: probability}.
    """
    # Group matchings by blocking count (since we select from minimum blocking count)
    blocking_groups = defaultdict(list)
    for matching_data in all_matchings:
        blocking_count = matching_data['blocking_count']
        blocking_groups[blocking_count].append(matching_data)

    # Find minimum blocking count
    min_blocking_count = min(blocking_groups.keys())
    best_matchings = blocking_groups[min_blocking_count]

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

                truthful_matchings = get_all_matchings_with_blocking_counts(truthful_combo)
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

                    strategic_matchings = get_all_matchings_with_blocking_counts(strategic_combo)
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