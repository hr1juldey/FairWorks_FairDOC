import os

def create_project_structure(base_path):
    """
    Creates the specified project directory structure, adds a filepath
    comment at the top of each created file, and includes enhancements
    for Windows path tolerance (e.g., long path support).

    Args:
        base_path (str): The root directory where the structure will be created.
                         For example, '/msp' or 'C:\\projects\\msp'.
    """

    # --- Helper for Windows long paths ---
    def _prepare_for_windows_os(path_str):
        """
        If on Windows, prepares an absolute path for OS calls by adding the
        \\\\?\\ prefix to potentially bypass MAX_PATH limits.
        The input path_str is expected to be already absolute and normalized.
        """
        if os.name == 'nt': # Check if OS is Windows ('nt' for NT-based Windows)
            # Ensure path is absolute, this should be guaranteed by prior processing
            # but as a safeguard:
            if not os.path.isabs(path_str):
                path_str = os.path.abspath(path_str)

            if not path_str.startswith("\\\\?\\"):
                if path_str.startswith("\\\\"):  # UNC path like \\server\share
                    return "\\\\?\\UNC\\" + path_str[2:]
                return "\\\\?\\" + path_str
        return path_str # Return as is for non-Windows OS or if already prefixed
    # --- End helper ---

    # Normalize and make the initial base_path absolute.
    # This ensures all subsequent paths derived from it will also be absolute.
    # The _prepare_for_windows_os will be applied just before OS calls.
    processed_base_path = os.path.abspath(os.path.normpath(base_path))

    # Define the structure as a dictionary
    structure = {
        "frontend": {
            "msp": {
                "main.py": None,
                "components": {
                    "__init__.py": None,
                    "chat_interface.py": None,
                    "question_flow.py": None,
                    "file_upload.py": None,
                    "case_report.py": None,
                },
                "mock_backend": {
                    "__init__.py": None,
                    "api_mock.py": None,
                    "nhs_data.py": None,
                },
                "styles": {
                    "__init__.py": None,
                    "material_theme.py": None,
                },
                "utils": {
                    "__init__.py": None,
                    "state_manager.py": None,
                },
            }
        }
    }

    # Helper function to recursively create directories and files
    def create_items(current_processed_path, items_dict):
        # current_processed_path is already absolute and normalized (but not \\?\ prefixed)
        for name, content in items_dict.items():
            # Construct the path for the current item
            item_path_segment = os.path.join(current_processed_path, name)
            # Normalize it (os.path.join on absolute current_path should be mostly fine)
            normalized_item_path = os.path.normpath(item_path_segment) # This is the user-friendly absolute path

            if content is None:  # It's a file
                try:
                    # Get the directory part for os.makedirs
                    parent_dir_path = os.path.dirname(normalized_item_path)
                    
                    # Prepare parent directory path for OS call if on Windows
                    path_for_os_parent_dir = _prepare_for_windows_os(parent_dir_path)
                    os.makedirs(path_for_os_parent_dir, exist_ok=True)

                    # Prepare file path for OS call if on Windows
                    path_for_os_file = _prepare_for_windows_os(normalized_item_path)
                    with open(path_for_os_file, 'w') as f:
                        # Comment uses the clean, normalized path
                        f.write(f"# File: {normalized_item_path}\n")
                    print(f"Created file: {normalized_item_path} (with filepath comment)")
                except OSError as e:
                    print(f"Error creating file {normalized_item_path}: {e}")
            else:  # It's a directory
                try:
                    # Prepare directory path for OS call if on Windows
                    path_for_os_dir = _prepare_for_windows_os(normalized_item_path)
                    os.makedirs(path_for_os_dir, exist_ok=True)
                    print(f"Created directory: {normalized_item_path}")
                    # Pass the clean normalized_item_path for recursion (it's already absolute)
                    create_items(normalized_item_path, content)
                except OSError as e:
                    print(f"Error creating directory {normalized_item_path}: {e}")

    # Start creating the structure from the processed (absolute, normalized) base path
    try:
        # Ensure the base directory itself exists, preparing for OS call
        path_for_os_base = _prepare_for_windows_os(processed_base_path)
        os.makedirs(path_for_os_base, exist_ok=True)
        print(f"Ensured base directory exists: {processed_base_path}")
    except OSError as e:
        print(f"Error creating base directory {processed_base_path}: {e}")
        return # Stop if base path cannot be created

    create_items(processed_base_path, structure)
    print("\nProject structure creation complete.")

if __name__ == "__main__":
    # IMPORTANT: Set your desired base path here.
    # For Windows, prefer absolute paths like 'C:\\Users\\YourUser\\my_msp_project_root'
    # For Linux/macOS, '/home/youruser/my_msp_project_root' or similar.
    #
    # The example below creates it in 'my_msp_project_root' in the current working directory.
    project_base_path = os.path.join(os.getcwd(), "my_msp_project_root")

    # If you truly intend to use a path like "/msp" (e.g., on Linux or if mapped on Windows):
    # project_base_path = "/msp"
    # Be cautious with system-level paths and ensure you have permissions.

    print(f"Attempting to create project structure at: {os.path.abspath(project_base_path)}")
    create_project_structure(project_base_path)

    # Display the expected structure using the normalized version of the base path
    final_display_base_path = os.path.abspath(os.path.normpath(project_base_path))
    print(f"\nTo verify, navigate to: {final_display_base_path}")
    print("The structure should be:")
    print(f"{final_display_base_path}{os.sep}") # Use os.sep for correct trailing slash
    print(f"└── frontend{os.sep}")
    print(f"    └── msp{os.sep}")
    print("        ├── main.py")
    print(f"        ├── components{os.sep}")
    print("        │   ├── __init__.py")
    print("        │   ├── chat_interface.py")
    print("        │   ├── question_flow.py")
    print("        │   ├── file_upload.py")
    print("        │   └── case_report.py")
    print(f"        ├── mock_backend{os.sep}")
    print("        │   ├── __init__.py")
    print("        │   ├── api_mock.py")
    print("        │   └── nhs_data.py")
    print(f"        ├── styles{os.sep}")
    print("        │   ├── __init__.py")
    print("        │   └── material_theme.py")
    print(f"        └── utils{os.sep}")
    print("            ├── __init__.py")
    print("            └── state_manager.py")
    print("\nEach .py file will now contain a comment at the top like '# File: /path/to/file.py'")
    print("On Windows, long path support has been enhanced.")
