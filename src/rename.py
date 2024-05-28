import os



def rename_extension(root_dir, path_ending=".stl.stl", path_renamed =".stl"):
    # List of elements in folder
    elements = os.listdir(root_dir)
    elements.sort()
    
    # For every element in the current folder
    for element in elements:
        path_to_element = root_dir + "/" + element
        if path_to_element[-8:] == ".stl.stl":
            os.rename(path_to_element, path_to_element[:-4])
        
        # Recursion over directories
        elif os.path.isdir(path_to_element):
            rename_extension(path_to_element)

path = "./data/Oriented_Divided_SH_L50_RECONSTRUCTED"

rename_extension(path)
