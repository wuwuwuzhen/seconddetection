import subprocess
import os
import shutil

def clear_directory_except_hash_file(dir_path, hash_file):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if filename == hash_file:
                continue  # 跳过哈希文件
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def get_current_commit_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

def generate_format_patch(start_commit=None, patch_dir='patch'):
    os.makedirs(patch_dir, exist_ok=True)
    if start_commit:
        subprocess.run(['git', 'format-patch', f'{start_commit}..HEAD', '--output-directory', patch_dir])
    else:
        subprocess.run(['git', 'format-patch', '-1', 'HEAD', '--output-directory', patch_dir])

def main():
    patch_dir = 'patch'
    hash_file_name = 'last_commit_hash.txt'
    hash_file_path = os.path.join(patch_dir, hash_file_name)
    
    # 清空patch目录，但保留哈希文件
    clear_directory_except_hash_file(patch_dir, hash_file_name)

    if os.path.exists(hash_file_path):
        with open(hash_file_path, 'r') as file:
            last_hash = file.read().strip()
            print(f"Generating format-patch from {last_hash} to HEAD in '{patch_dir}' directory")
            generate_format_patch(last_hash, patch_dir)
    else:
        print(f"Generating format-patch for the latest commit in '{patch_dir}' directory")
        generate_format_patch(patch_dir=patch_dir)

    # 保存当前的提交哈希
    current_hash = get_current_commit_hash()
    with open(hash_file_path, 'w') as file:
        file.write(current_hash)

if __name__ == "__main__":
    main()
