import os
import tarfile

def split_tar_by_number_of_files(source_tar_path, dest_dir, max_files=5000):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Open the source .tar file
    with tarfile.open(source_tar_path, 'r') as source_tar:
        file_count = 0
        picture_number = 1
        # Create the first output tar file
        out_tar_path = os.path.join(dest_dir, f'picture_{picture_number}.tar')
        out_tar = tarfile.open(out_tar_path, 'w')

        for member in source_tar.getmembers():
            # Add files to the output tar file until the max_files limit is reached
            if file_count < max_files:
                out_tar.addfile(member, source_tar.extractfile(member))
                file_count += 1
            else:
                # Close the current output tar file and create a new one
                out_tar.close()
                picture_number += 1
                out_tar_path = os.path.join(dest_dir, f'picture_{picture_number}.tar')
                out_tar = tarfile.open(out_tar_path, 'w')
                # Reset the file count and add the file to the new tar
                file_count = 1
                out_tar.addfile(member, source_tar.extractfile(member))

        # Close the last output tar file
        out_tar.close()

# Set the path to the large .tar file and the destination directory
source_tar_file_path = 'tar/picture.tar'  # Update this path
destination_directory = 'tar/'  # Update this path

split_tar_by_number_of_files(source_tar_file_path, destination_directory)

