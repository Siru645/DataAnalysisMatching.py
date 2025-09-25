def DA(StPrefs, SchPrefs):
    """
    Deferred Acceptance algorithm for 3 students and 3 schools

    Args:
        StPrefs: Dictionary mapping student roles ('a', 'b', 'c') to their preference lists
        SchPrefs: Dictionary mapping schools ('A', 'B', 'C') to their priority lists

    Returns:
        Dictionary mapping students to their assigned schools
    """
    # Define capacity of each school (1 in this case)
    Cpcty = {'X': 1, 'Y': 1, 'Z': 1}

    # Get list of students
    Students = list(StPrefs.keys())  # ['a', 'b', 'c']

    # For students who don't get matched, they'll be "matched" to themselves
    for i in StPrefs.keys():
        if i not in StPrefs[i]:  # Only add if not already there
            StPrefs[i].append(i)

    # Initialize tracking variables
    App = {}  # Application status: App[(school, student)] = 1 if student applied to school
    for i in Students:
        for j in SchPrefs.keys():
            App[(j, i)] = 0

    NumRej = {}  # Number of rejections each student has received
    for i in Students:
        NumRej[i] = 0

    # Track which students are still seeking a match
    Seeking = set(Students)
    Receiving = set()  # Schools receiving applications in current round
    ResStud = {}  # Final student assignments

    # Run DA algorithm
    while Seeking:
        # Students apply to their next choice
        for i in Seeking:
            # Get next school in preference list
            j = StPrefs[i][NumRej[i]]
            App[(j, i)] = 1
            Receiving.add(j)


        # Schools process applications
        for j in Receiving:
            # Get all applicants to this school
            Applicants = [i for i in Students if App[(j, i)] == 1]

            # If school has more applicants than capacity
            if len(Applicants) > Cpcty[j]:
                # Sort applicants by school's priority (worst first)
                sorted_applicants = []
                for p in SchPrefs[j][::-1]:  # Reverse order of priority list
                    if p in Applicants:
                        sorted_applicants.append(p) #this is the reversed order of applicants

                # Reject lowest priority applicants
                to_reject = sorted_applicants[:len(Applicants) - Cpcty[j]]
                for i in to_reject:
                    App[(j, i)] = 0
                    NumRej[i] += 1
                    # Add back to seeking if not at end of preference list
                    Seeking.add(i)

            # Remove tentatively accepted students from seeking
            for i in Applicants:
                if i in Seeking and App[(j, i)] == 1:
                    Seeking.remove(i)

        Receiving = set()

    # Construct final matching
    for i in Students:
        if i not in ResStud:
            # Student's final assignment is the school they didn't get rejected from
            assigned_school = StPrefs[i][NumRej[i]]
            ResStud[i] = assigned_school

    return ResStud