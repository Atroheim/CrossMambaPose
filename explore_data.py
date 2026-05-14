import h5py

def print_hdf5_structure(file_path):
    print(f"正在探测文件: {file_path}")
    print("-" * 50)
    
    with h5py.File(file_path, 'r') as f:
        # 这个函数会遍历文件里的每一个文件夹和变量
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📁 文件夹 (Group): {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"📄 变量 (Dataset): {name} | 形状: {obj.shape} | 类型: {obj.dtype}")
                
        f.visititems(print_attrs)
    print("-" * 50)

if __name__ == "__main__":
    radar_path = '/root/autodl-tmp/dataset/2022Jun25-0207/20220625020830/output_3D/keypoint3D_adjusted.npz'
    print_hdf5_structure(radar_path)