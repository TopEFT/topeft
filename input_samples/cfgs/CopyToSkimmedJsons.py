import os
import glob

# Pattern to match files that start with "ND_202" and end with .cfg
pattern = "ND_202*.cfg"

for old_filename in glob.glob(pattern):
    with open(old_filename, 'r') as infile:
        content = infile.read()

    # Replace inside the file:
    # - "ND_202" becomes "ND_skim202"
    # - ".json" becomes "_NDSkim.json"
    new_content = content.replace("ND_202", "ND_skim202").replace(".json", "_NDSkim.json")
    
    # Build the new file name by replacing the first occurrence of "ND_" with "NDSkim_"
    new_filename = old_filename.replace("ND_", "NDSkim_", 1)
    
    with open(new_filename, 'w') as outfile:
        outfile.write(new_content)
    
    print(f"Copied {old_filename} to {new_filename} with replacements.")
