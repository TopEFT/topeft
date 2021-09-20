import re

# Match strings using one or more regular expressions
def regex_match(lst,regex_lst):
    # NOTE: For the regex_lst patterns, we use the raw string to generate the regular expression.
    #       This means that any regex special characters in the regex_lst should be properly
    #       escaped prior to calling this function.
    # NOTE: The input list is assumed to be a list of str objects and nothing else!
    matches = []
    if len(regex_lst) == 0: return lst[:]
    for s in lst:
        for pat in regex_lst:
            m = re.search(r"{}".format(pat),s)
            if m is not None:
                matches.append(s)
                break
    return matches