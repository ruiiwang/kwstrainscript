import pickle
import numpy as np

with open('converted_8/converted_HeyMemo_features.pkl', 'rb') as f:
    data = pickle.load(f)

# 查看总项数
print(f"总项数: {len(data)}")
print(data[0].shape)
print(data[1].shape)
print(data[1])

# 查看每一项的形状
# for i, item in enumerate(data):
#     if isinstance(item, (list, np.ndarray)):
#         print(f"第{i}项: 形状{len(item)}")
#     elif isinstance(item, dict):
#         print(f"第{i}项: 字典键{list(item.keys())}")
#     else:
#         print(f"第{i}项: 类型{type(item)}")
