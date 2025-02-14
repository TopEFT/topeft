import os
import json

# Base path to check the existence of files
BASE_PATH = "/cms/cephfs/data/"

# Function to update JSON files by removing missing files
def update_json_files(directory, base_path):
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):
            json_path = os.path.join(directory, file_name)
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                if "files" in data:
                    # Identify missing files
                    existing_files = []
                    missing_files = []
                    for rel_path in data["files"]:
                        full_path = os.path.join(base_path, rel_path.lstrip('/'))
                        if os.path.exists(full_path):
                            existing_files.append(rel_path)
                        else:
                            missing_files.append(rel_path)

                    # Update JSON if there are missing files
                    if missing_files:
                        print(f"Updating {file_name}: {len(missing_files)} missing files removed.")
                        data["files"] = existing_files
                        with open(json_path, 'w') as json_output:
                            json.dump(data, json_output, indent=4)

# Main execution
if __name__ == "__main__":
    current_directory = os.getcwd()
    print(f"Working in directory: {current_directory}")
    update_json_files(current_directory, BASE_PATH)
    print("Processing complete.")
