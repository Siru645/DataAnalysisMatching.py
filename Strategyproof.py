import random
import pandas as pd

# Load the data
data = pd.read_csv(
    r'C:\Users\lsr64\OneDrive - Georgia State University\Georgia State University\My Research\Projects\Matching with incomplete preferences\Data analysis\Book1.csv')

def parse_preference_lists(pref_string):
    """
    Parse preference lists from string format.
    Args:
        pref_string: String like "X,Y,Z" or "X,Y,Z;Y,X,Z;Z,Y,X"
    Returns:
        List of preference lists, where each list contains school names
    Example: Input: "X,Y,Z;Y,X,Z;Z,Y,X" Output: [['X', 'Y', 'Z'], ['Y', 'X', 'Z'], ['Z', 'Y', 'X']]
    """
    if pd.isna(pref_string) or pref_string.strip() == '':
        return []

    # Split by semicolon if multiple lists exist
    if ';' in pref_string:
        lists = pref_string.split(';')
    else:
        lists = [pref_string]

    # Clean up each list and convert to list of schools
    parsed_lists = []
    for pref_list in lists:
        schools = [school.strip() for school in pref_list.split(',')]
        parsed_lists.append(schools)

    return parsed_lists


def load_student_preferences():
    """
    Load and organize all student preference lists from ALL rows in the CSV data.
    Keeps preference lists grouped by row.

    Returns:
        Dictionary with organized preference data, grouped by row
    """
    preferences = {
        'student_a': {
            'true_lists': [],
            'submitted_lists': []
        },
        'student_b': {
            'submitted_lists': []
        },
        'student_c': {
            'submitted_lists': []
        }
    }

    # Process each row in the CSV
    for row_index in range(len(data)):
        # Student A true lists
        a_true = data['student a true lists'].iloc[row_index]
        # Keep the parsed result as a group
        parsed_lists = parse_preference_lists(a_true)
        preferences['student_a']['true_lists'].append(parsed_lists)

        # Student A submitted lists
        a_submitted = data['student a submitted lists'].iloc[row_index]
        parsed_lists = parse_preference_lists(a_submitted)
        preferences['student_a']['submitted_lists'].append(parsed_lists)

        # Student B submitted lists
        b_submitted = data['student b submitted lists'].iloc[row_index]
        parsed_lists = parse_preference_lists(b_submitted)
        preferences['student_b']['submitted_lists'].append(parsed_lists)

        # Student C submitted lists
        c_submitted = data['student c submitted lists'].iloc[row_index]
        parsed_lists = parse_preference_lists(c_submitted)
        preferences['student_c']['submitted_lists'].append(parsed_lists)

    return preferences

student_preferences = load_student_preferences()
print(student_preferences)


def get_bc_combinations(student_preferences):
    """
    Generate all possible combinations of student B and C preference list groups.
    Each combination fixes one cell from B and one cell from C.

    Args:
        student_preferences: Dictionary from load_student_preferences()

    Returns:
        List of dictionaries, each containing one B-C combination
    """
    b_groups = student_preferences['student_b']['submitted_lists']
    c_groups = student_preferences['student_c']['submitted_lists']

    bc_combinations = []

    for b_index, b_group in enumerate(b_groups):
        for c_index, c_group in enumerate(c_groups):
            bc_combinations.append({
                'b_group_index': b_index,
                'c_group_index': c_index,
                'b_preference_lists': b_group,  # This is a list of lists from one cell
                'c_preference_lists': c_group  # This is a list of lists from one cell
            })

    return bc_combinations


def run_matching_mechanism(players, mechanism):
    """
    Run a matching mechanism (DA or TTC) with multiple preference lists for each student,
    trying all possible combinations of preference lists.

    Args:
        players: List of player objects
        mechanism: Function reference to the matching algorithm (DA or TTC)

    Returns:
        Dictionary mapping students to their assigned schools after selection
    """
    import copy

    # Step 1: Extract student roles and their multiple preference lists
    student_pref_lists = {}
    for player in players:
        # Parse the string-formatted preference lists
        pref_lists_str = player.preference_lists

        # Split by semicolon if multiple lists exist
        if ';' in pref_lists_str:
            lists = pref_lists_str.split(';')
        else:
            lists = [pref_lists_str]

        student_pref_lists[player.student_role] = lists

    print(f"student_pref_lists: {student_pref_lists}")

    # Step 2: Prepare school preferences (fixed for all  mechanisms)
    SchPrefs = {
        'X': ['a', 'c', 'b'],
        'Y': ['b', 'a', 'c'],
        'Z': ['c', 'b', 'a']
    }

    # Step 3: Generate all possible combinations of preference lists
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

    print(f"Generated {len(all_combinations)} combinations")

    # Step 4: Run the matching mechanism for each combination
    all_matchings = []
    for combo_idx, combo in enumerate(all_combinations):
        print(f"Processing combination {combo_idx + 1}/{len(all_combinations)}: {combo}")

        # Format for the matching mechanism
        StPrefs = {}
        for role, preferences_str in combo.items():
            # Convert comma-separated string to list, ensuring it matches expected format
            preferences_list = preferences_str.split(',')
            # Make sure each item in the list is stripped of whitespace
            preferences_list = [p.strip() for p in preferences_list]
            StPrefs[role] = preferences_list

        # Create deep copies for the mechanism to avoid modifying the originals
        StPrefs_copy = copy.deepcopy(StPrefs)
        SchPrefs_copy = copy.deepcopy(SchPrefs)

        print(f"Calling mechanism with StPrefs: {StPrefs_copy}")
        print(f"Calling mechanism with SchPrefs: {SchPrefs_copy}")

        try:
            # Run the mechanism and store results
            result = mechanism(StPrefs_copy, SchPrefs_copy)
            print(f"Mechanism result: {result}")

            all_matchings.append({
                'preferences': combo,
                'matching': result
            })
        except Exception as e:
            print(f"Error running mechanism: {str(e)}")
            import traceback
            traceback.print_exc()

    # Check if we have any matchings
    if not all_matchings:
        print("No matchings were generated!")
        # Create a default matching
        default_matching = {role: 'No match' for role in student_roles}
        return default_matching

    # Step 5: Select the final matching based on the mechanism
    final_matching_data = select_final_matching(all_matchings, mechanism, student_pref_lists)
    final_matching = final_matching_data['matching']
    selected_preferences = final_matching_data['preferences']

    print(f"Final matching selected: {final_matching}")
    print(f"Selected preferences: {selected_preferences}")

    # Step 6: Assign results to players
    for player in players:
        player.assigned_school = final_matching.get(player.student_role, "No match")
        player.selected_preference_list = selected_preferences.get(player.student_role, "")
        print(
            f"Assigned {player.student_role} to {player.assigned_school} with preference list {player.selected_preference_list}")

    return final_matching

def select_DA_matching(all_matchings, student_pref_lists):
    """
    Select the final matching for DA mechanism based on the number of students who would like to swap.
    The matching with the minimal number of students who would like to swap is selected.
    If multiple matchings have the same minimal number, one is randomly chosen.

    Args:
        all_matchings: List of dictionaries containing preferences and matchings
        student_pref_lists: Dictionary mapping students to their multiple preference lists

    Returns:
        Dictionary with matching and preferences used
    """
    import random

    if not all_matchings:
        return {'matching': {}, 'preferences': {}}  # Return empty dictionaries if no matchings are available

    # Track the matchings with the minimum number of students who want to swap
    min_swap_count = float('inf')
    best_matchings = []

    for matching_data in all_matchings:
        matching = matching_data['matching']
        preferences = matching_data['preferences']

        # Count how many students would like to swap in this matching
        # Consider ALL preference lists, not just the one used in this matching
        swap_count = count_students_willing_to_swap(matching, student_pref_lists)


        if swap_count < min_swap_count:
            min_swap_count = swap_count
            best_matchings = [{'matching': matching, 'preferences': preferences}]
        elif swap_count == min_swap_count:
            best_matchings.append({'matching': matching, 'preferences': preferences})

    # Randomly choose one of the best matchings
    chosen_matching = random.choice(best_matchings)

    return chosen_matching

def count_students_willing_to_swap(matching, student_pref_lists):
    """
    Count how many students would like to swap their allocations.
    A group of students is willing to swap if they all prefer another student's
    school to their own based on ALL of their preference lists.

    Args:
        matching: Dictionary mapping students to their assigned schools
        student_pref_lists: Dictionary mapping students to all their preference lists

    Returns:
        Number of students willing to swap (0, 2, or 3)
    """
    students = list(matching.keys())
    if len(students) != 3:
        return 0  # Not enough students for a swap

    # Parse all preference lists for each student
    parsed_pref_lists = {}
    for student_role, pref_lists in student_pref_lists.items():
        parsed_pref_lists[student_role] = []
        for pref_str in pref_lists:
            # Split and strip each preference list
            parsed_list = [school.strip() for school in pref_str.split(',')]
            parsed_pref_lists[student_role].append(parsed_list)

    # Check for a 3-way swap (cycle)
    # We'll check all possible 3-way cycles without hardcoding roles

    # 3-way swaps require checking all permutations of 3 students
    # There are 2 possible cycle directions with 3 students

    # Check for any valid 3-way swap
    if check_three_way_swap(students, matching, parsed_pref_lists):
        return 3  # All three students prefer a 3-way swap

    # Check for any valid 2-way swap
    if check_two_way_swap(students, matching, parsed_pref_lists):
        return 2  # Two students prefer to swap

    return 0  # No students willing to swap

def check_three_way_swap(students, matching, parsed_pref_lists):
    """
    Check if all three students would prefer a 3-way swap in ALL their preference lists.

    Args:
        students: List of student roles
        matching: Dictionary mapping students to their assigned schools
        parsed_pref_lists: Dictionary mapping students to their parsed preference lists

    Returns:
        True if a valid 3-way swap exists, False otherwise
    """
    # Try both possible cycle directions

    # Forward cycle: students[0] → students[1] → students[2] → students[0]
    forward_cycle = (
            all(prefers_school_over_another(pref_list, matching[students[1]], matching[students[0]])
                for pref_list in parsed_pref_lists[students[0]]) and
            all(prefers_school_over_another(pref_list, matching[students[2]], matching[students[1]])
                for pref_list in parsed_pref_lists[students[1]]) and
            all(prefers_school_over_another(pref_list, matching[students[0]], matching[students[2]])
                for pref_list in parsed_pref_lists[students[2]])
    )

    # Backward cycle: students[0] → students[2] → students[1] → students[0]
    backward_cycle = (
            all(prefers_school_over_another(pref_list, matching[students[2]], matching[students[0]])
                for pref_list in parsed_pref_lists[students[0]]) and
            all(prefers_school_over_another(pref_list, matching[students[0]], matching[students[1]])
                for pref_list in parsed_pref_lists[students[1]]) and
            all(prefers_school_over_another(pref_list, matching[students[1]], matching[students[2]])
                for pref_list in parsed_pref_lists[students[2]])
    )

    return forward_cycle or backward_cycle

def check_two_way_swap(students, matching, parsed_pref_lists):
    """
    Check if any two students would prefer to swap schools in ALL their preference lists.

    Args:
        students: List of student roles
        matching: Dictionary mapping students to their assigned schools
        parsed_pref_lists: Dictionary mapping students to their parsed preference lists

    Returns:
        True if a valid 2-way swap exists, False otherwise
    """
    # Check all possible pairs of students
    for i in range(len(students)):
        for j in range(i + 1, len(students)):
            student1 = students[i]
            student2 = students[j]

            # Check if both students prefer to swap in ALL their preference lists
            both_prefer_swap = (
                    all(prefers_school_over_another(pref_list, matching[student2], matching[student1])
                        for pref_list in parsed_pref_lists[student1]) and
                    all(prefers_school_over_another(pref_list, matching[student1], matching[student2])
                        for pref_list in parsed_pref_lists[student2])
            )

            if both_prefer_swap:
                return True

    return False

def prefers_school_over_another(preference_list, school1, school2):
    """
    Check if a student prefers school1 over school2 based on their preference list.

    Args:
        preference_list: List of schools in order of preference
        school1: First school to compare
        school2: Second school to compare

    Returns:
        True if the student prefers school1 over school2, False otherwise
    """
    try:
        rank1 = preference_list.index(school1)
        rank2 = preference_list.index(school2)
        return rank1 < rank2  # Lower index means higher preference
    except ValueError:
        # If either school is not in the list, assume they don't prefer school1
        return False


