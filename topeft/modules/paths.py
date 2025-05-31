import os
import topeft

pjoin = os.path.join

# This function takes as input any path (inside of topeft/topeft), and returns the absolute path
def topeft_path(path_in_repo):
    return pjoin(topeft.__path__[0], path_in_repo)
