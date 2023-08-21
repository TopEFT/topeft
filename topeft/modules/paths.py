import os
import topcoffea

pjoin = os.path.join

# This function takes as input any path (inside of topcoffea/topcoffea), and returns the absolute path
def topcoffea_path(path_in_repo):
    return pjoin(topcoffea.__path__[0], path_in_repo)
