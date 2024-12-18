import os
import json

# Base path to check the existence of files
BASE_PATH = "/cms/cephfs/data/"

# Function to check file existence
def check_files_in_json(directory, base_path):
    results = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):
            json_path = os.path.join(directory, file_name)
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                if "files" in data:
                    missing_files = []
                    for rel_path in data["files"]:
                        full_path = os.path.join(base_path, rel_path.lstrip('/'))
                        if not os.path.exists(full_path):
                            missing_files.append(full_path)
                    results[file_name] = {
                        "total_files": len(data["files"]),
                        "missing_files": missing_files,
                        "missing_count": len(missing_files),
                    }
    return results

# Main execution
if __name__ == "__main__":
    current_directory = os.getcwd()
    results = check_files_in_json(current_directory, BASE_PATH)

    for json_file, result in results.items():
        print(f"JSON File: {json_file}")
        print(f"  Total Files: {result['total_files']}")
        print(f"  Missing Files: {result['missing_count']}")
        if result['missing_files']:
            print(f"  Missing Files Sample: {result['missing_files'][:5]}")
        print()
