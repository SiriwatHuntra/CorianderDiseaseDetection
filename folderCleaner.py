import os
import glob

def clear_folders(folders):
    for folder in folders:
        files = glob.glob(f"{folder}/*")
        for file in files:
            try:
                os.remove(file)
                print(f"File {file} has been removed successfully")
            except Exception as e:
                print(f"Error occurred while deleting file {file}. Error message: {str(e)}")


#usage
#output_folders = ["classed/h", "classed/u", "classed/n", "classed/w"]
#clear_folders(output_folders)
