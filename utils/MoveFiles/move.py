import shutil
import os

def main(models):

    folder = 'outputs//' + models

    path = os.path.join(os.getcwd(), folder)

    # fixed paths    
    dest1 = "C://Users//Nikolaus Kresse//OneDrive//MasterArbeitLatex//figures//Performance"
    dest2 = "C://Users//Nikolaus Kresse//OneDrive//MasterArbeitLatex//figures//Sharps"


    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    filtered_folders = [folder for folder in folders if "models" not in folder]


    def find_files_in_directory(directory, keyword):
        matching_files = {}
        for root, _, files in os.walk(directory):
            for filename in files:
                if keyword in filename:
                    full_path = os.path.join(root, filename)
                    # Use the final filename as the key in the dictionary
                    final_filename = os.path.basename(filename)
                    matching_files[final_filename] = full_path
        return matching_files

    # Search for files containing "Sharpe_ratio" and "Performance" in subdirectories
    sharpe_ratio_files = find_files_in_directory(path, "Sharpe_ratio")
    performance_files = find_files_in_directory(path, "Performance")

    def move_files(files, dest):
        for filename, path in files.items():
            try:            
                shutil.copy(path, os.path.join(dest, filename))
                print(f'Successfully moved {filename} to {dest}')
            except Exception as e:
                print(e)                                    


    move_files(performance_files, dest=dest1)
    move_files(sharpe_ratio_files, dest=dest2)

if __name__ == "__main__":  
    models = '20230907-133040' 
    main(models)                
