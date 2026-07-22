import os

def check_data_structure(data_dir='data'):
    """Checks if the data directory has the required 'yes' and 'no' folders."""
    required_folders = ['yes', 'no']
    missing = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return False
        
    for folder in required_folders:
        path = os.path.join(data_dir, folder)
        if not os.path.exists(path):
            missing.append(folder)
        else:
            count = len(os.listdir(path))
            print(f"Found '{folder}' folder with {count} images.")
            
    if missing:
        print(f"Missing folders in '{data_dir}': {missing}")
        print("\nPlease structure your data as follows:")
        print("data/")
        print("├── yes/  (Brain images WITH tumor)")
        print("└── no/   (Brain images WITHOUT tumor)")
        return False
        
    print("\nData structure is correct!")
    return True

if __name__ == "__main__":
    check_data_structure()
