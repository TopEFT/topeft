import os
import json

# Directory containing your JSON files
directory = "./"  # change if needed
prefix_to_remove = "/cms/cephfs/data"

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping {filename} (invalid JSON)")
                continue

        # Recursively remove prefix from any string that starts with it
        def remove_prefix(obj):
            if isinstance(obj, str) and obj.startswith(prefix_to_remove):
                return obj[len(prefix_to_remove):]  # cut off prefix
            elif isinstance(obj, list):
                return [remove_prefix(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: remove_prefix(v) for k, v in obj.items()}
            return obj

        new_data = remove_prefix(data)

        # Overwrite JSON file with updated content
        with open(filepath, "w") as f:
            json.dump(new_data, f, indent=4)

        print(f"Processed {filename}")

