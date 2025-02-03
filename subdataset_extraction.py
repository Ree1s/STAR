import os
import pandas as pd

# 配置路径
video_dir = "dataset/OpenVid-1M/video"
csv_path = "dataset/OpenVid-1M/data/train/OpenVid-1M.csv"
output_csv = "dataset/OpenVid-1M/data/train/OpenVid-1M_subset.csv"

# 读取原始CSV文件
df = pd.read_csv(csv_path)

# 假设CSV中包含视频文件名的列名为"video_name"（根据实际情况修改）
# 如果文件名不带扩展名，可以添加对应的视频格式后缀
df['video_exists'] = df['video'].apply(
    lambda x: os.path.exists(os.path.join(video_dir, x))
)

# 筛选出存在视频文件的记录
subset_df = df[df['video_exists']].copy().drop(columns=['video_exists'])

# 保存结果
subset_df.to_csv(output_csv, index=False)
print(f"生成小CSV完成，有效记录数: {len(subset_df)}/{len(df)}")