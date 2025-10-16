import pickle
import numpy as np
import os

folder = 'converted_2'
pkl_files = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith('.pkl')]

for pkl_path in pkl_files:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(pkl_path)
    print(f"总项数: {len(data)}")
    print(data[0].shape)
    print(data[1].shape)
    print(data[1])

# with open('converted_un/1.6s_half_augment_data.pkl', 'rb') as f:
#     data = pickle.load(f)

# # 查看总项数
# print(f"总项数: {len(data)}")
# print(data[0].shape)
# print(data[1].shape)
# print(data[1])

# 查看每一项的形状
# for i, item in enumerate(data):
#     if isinstance(item, (list, np.ndarray)):
#         print(f"第{i}项: 形状{len(item)}")
#     elif isinstance(item, dict):
#         print(f"第{i}项: 字典键{list(item.keys())}")
#     else:
#         print(f"第{i}项: 类型{type(item)}")
