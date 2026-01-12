
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
import h5py
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# --- 1. 配置区域 (请根据您的需求修改) ---
# =============================================================================
DATA_DIR = ''
SESSION_IDS = ['']
IMU_FILE_TEMPLATE = '{session_id}_IMU.txt'
EMG_FILE_TEMPLATE = '{session_id}_EMG.txt'
GROUND_TRUTH_FILE_TEMPLATE = '{session_id}_gt.txt'
WINDOW_SIZE_S = 0.5
WINDOW_STEP_S = 0.1
PROCESSED_DATA_OUTPUT = 'processed_sensor_data.h5'
LABEL_MAP_OUTPUT = 'action_label_map.json'
RAW_TIMESERIES_VIS_OUTPUT = 'interactive_raw_timeseries_plot.html'
TIMESERIES_VIS_OUTPUT = 'interactive_normalized_timeseries_plot.html'
FEATURE_VIS_OUTPUT = 'interactive_feature_plot.html'
SEPARABILITY_REPORT_OUTPUT = 'separability_analysis_report.txt'
MERGED_FEATURE_DATA_OUTPUT = 'merged_feature_data.h5'
MERGED_FEATURE_VIS_OUTPUT = 'interactive_merged_feature_plot.html'

ACTION_LABELS = {
    "站立": 0, "行走": 1,
    "坐下": 2, "从站到坐": 3, "从坐到站": 4,
    "深蹲": 5, "从站到蹲": 6, "从蹲到站": 7,
    "左转": 8, "右转": 9
}

LABEL_COLORS = [
    'rgba(31, 119, 180, 0.2)',  # 0: 站立 (Blue)
    'rgba(255, 127, 14, 0.2)',  # 1: 行走 (Orange)
    'rgba(44, 160, 44, 0.2)',   # 2: 坐下 (Green)
    'rgba(214, 39, 40, 0.2)',   # 3: 从站到坐 (Red)
    'rgba(148, 103, 189, 0.2)', # 4: 从坐到站 (Purple)
    'rgba(140, 86, 75, 0.2)',   # 5: 深蹲 (Brown)
    'rgba(227, 119, 194, 0.2)', # 6: 从站到蹲 (Pink)
    'rgba(127, 127, 127, 0.2)', # 7: 从蹲到站 (Gray)
    'rgba(188, 189, 34, 0.2)',  # 8: 左转 (Olive)
    'rgba(23, 190, 207, 0.2)'   # 9: 右转 (Cyan)
]


# =============================================================================
# --- 2. 核心功能函数 ---
# =============================================================================

def parse_time_from_line(line):
    match = re.search(r'time: ([\d:.]+)', line)
    return match.group(1) if match else None

def parse_emg_data(file_path):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            time_str = parse_time_from_line(line)
            if not time_str: continue
            data = {'time': time_str}
            parts = line.split()
            for i in range(0, len(parts) - 2, 2):
                try:
                    data[parts[i].replace(':', '')] = float(parts[i+1])
                except (ValueError, IndexError):
                    continue
            records.append(data)
    df = pd.DataFrame(records)
    if 'time' not in df.columns: return pd.DataFrame()
    df['datetime'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['datetime']).set_index('datetime').drop(columns=['time'])
    df = df.groupby(df.index).mean()
    return df.astype(float)

def parse_imu_data(file_path):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            time_str = parse_time_from_line(line)
            if not time_str: continue
            
            data = {'time': time_str}
            current_imu_prefix = ""
            parts = line.split()
            
            i = 0
            while i < len(parts) - 1:
                key = parts[i].replace(':', '')
                value_str = parts[i+1]
                
                # 检查是否是新的IMU组的开始
                if key.startswith('IMU'):
                    current_imu_prefix = key.split('_')[0].replace('__', '_') + '_' # e.g., 'IMU1_'
                    data_key = key.replace('__', '_')
                    data[data_key] = float(value_str)
                elif key in ['AX', 'AY', 'AZ', 'VX', 'VY', 'VZ']:
                    # 使用当前IMU前缀构建完整的列名
                    if current_imu_prefix:
                        full_key = current_imu_prefix + key
                        data[full_key] = float(value_str)
                # 跳过时间戳键值对，因为它已被预处理
                elif key == 'time':
                    i += 1 # time 后面只有一个值
                    continue
                
                i += 2 # 步进2以处理下一个键值对
            records.append(data)

    df = pd.DataFrame(records)
    if 'time' not in df.columns: return pd.DataFrame()
    df['datetime'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['datetime']).set_index('datetime').drop(columns=['time'])
    df = df.groupby(df.index).mean()
    return df

def parse_ground_truth(file_path, label_map):
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3 and parts[2] in label_map:
                records.append({
                    'start_time': pd.to_datetime(parts[0], format='%H:%M:%S.%f'),
                    'end_time': pd.to_datetime(parts[1], format='%H:%M:%S.%f'),
                    'label': label_map[parts[2]], 'label_name': parts[2]
                })
    return pd.DataFrame(records)

def calculate_joint_kinematics(imu_df):
    kinematics_df = pd.DataFrame(index=imu_df.index)

    # 计算髋关节和膝关节数据，并将结果从弧度转换为度
    if 'IMU1_AX' in imu_df.columns:
        # 原始数据是弧度，直接转换为度
        kinematics_df['髋关节角度(矢状面)'] = np.rad2deg(imu_df['IMU1_AX']) + 80  # 假设IMU1_AX是髋关节的角度数据，且需要加80度偏移
    if 'IMU1_AX' in imu_df.columns and 'IMU2_AX' in imu_df.columns:
        # 弧度相减后，再转换为度
        knee_angle_rad = imu_df['IMU1_AX'] - imu_df['IMU2_AX']
        kinematics_df['膝关节角度(矢状面)'] = np.rad2deg(knee_angle_rad) + 180
    if 'IMU2_AX' in imu_df.columns and 'IMU3_AX' in imu_df.columns:
        # 弧度相减后，再转换为度
        ankle_angle_rad = imu_df['IMU2_AX'] - imu_df['IMU3_AX']
        kinematics_df['踝关节角度(矢状面)'] = np.rad2deg(ankle_angle_rad) + 10

    # 基于“度”为单位的角度计算角速度 (单位: 度/秒)
    dt = np.mean(np.diff(imu_df.index.values)) / np.timedelta64(1, 's')
    if dt > 0:
        if '髋关节角度(矢状面)' in kinematics_df.columns:
            kinematics_df['髋关节角速度(矢状面)'] = np.gradient(kinematics_df['髋关节角度(矢状面)'], dt)
        if '膝关节角度(矢状面)' in kinematics_df.columns:
            kinematics_df['膝关节角速度(矢状面)'] = np.gradient(kinematics_df['膝关节角度(矢状面)'], dt)
        if '踝关节角度(矢状面)' in kinematics_df.columns:
            kinematics_df['踝关节角速度(矢状面)'] = np.gradient(kinematics_df['踝关节角度(矢状面)'], dt)
    
    return kinematics_df.fillna(0)


def extract_features(data_df, label_col, fs, window_size_s, window_step_s):
    window_length = int(window_size_s * fs)
    step_length = int(window_step_s * fs)
    features, labels = [], []
    feature_cols = [col for col in data_df.columns if col != label_col]
    
    for i in range(0, len(data_df) - window_length, step_length):
        window = data_df[feature_cols].iloc[i : i + window_length]
        window_labels = data_df[label_col].iloc[i : i + window_length]
        if window_labels.empty: continue
        window_label = window_labels.mode()[0]
        
        mav = window.abs().mean()
        rms = np.sqrt((window**2).mean())
        wl = window.diff().abs().sum()
        
        f, Pxx = welch(window, fs, nperseg=window_length, axis=0)
        total_power = np.sum(Pxx, axis=0)
        non_zero_power_mask = total_power > 1e-10
        mnf = np.zeros_like(total_power)
        numerator = np.sum(f[:, np.newaxis] * Pxx, axis=0)
        mnf[non_zero_power_mask] = numerator[non_zero_power_mask] / total_power[non_zero_power_mask]

        cumsum_pxx = np.cumsum(Pxx, axis=0)
        half_power = total_power / 2.0
        mdf_indices = np.zeros(Pxx.shape[1], dtype=int)
        for col_idx in range(Pxx.shape[1]):
            if non_zero_power_mask[col_idx]:
                mdf_indices[col_idx] = np.searchsorted(cumsum_pxx[:, col_idx], half_power[col_idx])
        
        mdf_indices = np.clip(mdf_indices, 0, len(f) - 1)
        mdf = f[mdf_indices]

        feat_row = {f'{col}_mav': m for col, m in mav.items()}
        feat_row.update({f'{col}_rms': r for col, r in rms.items()})
        feat_row.update({f'{col}_wl': w for col, w in wl.items()})
        feat_row.update({f'{col}_mnf': mn for col, mn in zip(window.columns, mnf)})
        feat_row.update({f'{col}_mdf': md for col, md in zip(window.columns, mdf)})
        
        features.append(feat_row)
        labels.append(window_label)

    feature_df = pd.DataFrame(features)
    feature_df['label'] = labels
    return feature_df.dropna()


def visualize_raw_time_series(df, gt_df, output_path):
    print(f"正在生成归一化前的交互式时间序列图表: {output_path}")
    fig = make_subplots(rows=len(df.columns), cols=1, shared_xaxes=True, subplot_titles=df.columns)
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode='lines'), row=i+1, col=1)
    shapes = []

    for _, row in gt_df.iterrows():
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=row['start_time'], y0=0,
                           x1=row['end_time'], y1=1, fillcolor=LABEL_COLORS[row['label']],
                           layer="below", line_width=0))
    fig.update_layout(title='交互式原始传感器数据可视化 (归一化前)',
                      height=200 * len(df.columns), shapes=shapes, showlegend=True)
    fig.write_html(output_path)

def visualize_time_series(df, gt_df, output_path):
    print(f"正在生成归一化后的交互式时间序列图表: {output_path}")
    fig = make_subplots(rows=len(df.columns), cols=1, shared_xaxes=True, subplot_titles=df.columns)
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode='lines'), row=i+1, col=1)
    shapes = []
    for _, row in gt_df.iterrows():
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=row['start_time'], y0=0,
                           x1=row['end_time'], y1=1, fillcolor=LABEL_COLORS[row['label']],
                           layer="below", line_width=0))
    fig.update_layout(title='交互式归一化传感器数据可视化 (可点击图例隐藏/显示曲线)',
                      height=200 * len(df.columns), shapes=shapes, showlegend=True)
    fig.write_html(output_path)

def visualize_merged_data(df, gt_df, output_path):
    print(f"正在生成合并数据的交互式图表: {output_path}")
    
    # 选择一些有代表性的列来避免图表过于混乱
    raw_cols_to_plot = [c for c in ['EMG1', 'IMU1_AX', '膝关节角度(矢状面)'] if c in df.columns]
    feature_cols_to_plot = [c for c in ['EMG1_mav', 'IMU1_AX_mav', '膝关节角度(矢状面)_mav'] if c in df.columns]
    
    if not raw_cols_to_plot or not feature_cols_to_plot:
        print("警告: 无法找到足够的代表性列来进行合并数据可视化，已跳过。")
        return
        
    num_rows = len(raw_cols_to_plot)
    # 为每个原始信号创建一个子图，并在同一个子图中绘制其对应的特征
    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, 
                        subplot_titles=[f"{col} vs {col}_mav" for col in raw_cols_to_plot])

    for i, (raw_col, feat_col) in enumerate(zip(raw_cols_to_plot, feature_cols_to_plot)):
        # 绘制原始信号
        fig.add_trace(go.Scatter(x=df.index, y=df[raw_col], name=raw_col, mode='lines',
                                 legendgroup=f'group{i}', line=dict(width=1.5)), row=i+1, col=1)
        # 绘制对应的特征信号
        fig.add_trace(go.Scatter(x=df.index, y=df[feat_col], name=feat_col, mode='lines',
                                 legendgroup=f'group{i}', line=dict(width=2, dash='dash')), row=i+1, col=1)

    # 添加动作标签的背景高亮区域
    shapes = []
    for _, row in gt_df.iterrows():
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=row['start_time'], y0=0,
                           x1=row['end_time'], y1=1, fillcolor=LABEL_COLORS[row['label']],
                           layer="below", line_width=0))

    fig.update_layout(title='原始信号与对应窗口特征的可视化对比',
                      height=250 * num_rows, shapes=shapes, showlegend=True)
    fig.write_html(output_path)


def analyze_and_visualize_features(feature_df, id_to_name_map, report_path, vis_path):
    print("正在进行特征分离度评估...")
    if feature_df.shape[0] < 20:
        print("特征数量过少，跳过分离度评估。")
        return
    X = feature_df.drop('label', axis=1)
    y = feature_df['label']
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(clf, X, y, cv=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report_str = f"--- 动作可分离性评估报告 ---\n\n5折交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n"
    target_names = [id_to_name_map[i] for i in sorted(y.unique())]
    report_str += "详细分类报告 (在30%的测试集上):\n" + classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    report_str += "\n\n混淆矩阵:\n" + pd.DataFrame(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())), index=target_names, columns=target_names).to_string()
    print(report_str)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_str)
    print(f"评估报告已保存到: {report_path}")

    print(f"正在生成特征分布3D可视化图表: {vis_path}")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['label_name'] = y.map(id_to_name_map).values
    fig = go.Figure()
    for name, group in pca_df.groupby('label_name'):
        fig.add_trace(go.Scatter3d(x=group['PC1'], y=group['PC2'], z=group['PC3'],
                                   mode='markers', name=name, marker=dict(size=5, opacity=0.8)))
    fig.update_layout(title='特征PCA降维3D可视化', margin=dict(l=0, r=0, b=0, t=40),
                      scene=dict(xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                 yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                                 zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'))
    fig.write_html(vis_path)

# =============================================================================
# --- 3. 主执行流程 ---
# =============================================================================
def main():
    global SESSION_IDS
    if not SESSION_IDS:
        sessions = set(re.match(r'(people\d+_\d+)_', f).group(1) for f in os.listdir(DATA_DIR) if re.match(r'(people\d+_\d+)_', f))
        SESSION_IDS = sorted(list(sessions))
        if not SESSION_IDS:
            print(f"错误：在目录 '{DATA_DIR}' 中未找到任何 'peopleX_Y' 格式的数据文件。")
            return
    print(f"发现 {len(SESSION_IDS)} 个采集会话: {SESSION_IDS}")

    all_processed_data = {}
    all_raw_data = {} # 【V6 新增】用于存储归一化前的数据
    session_sampling_rates = {}

    for session_id in SESSION_IDS:
        print(f"\n--- 正在处理会话: {session_id} ---")
        paths = [os.path.join(DATA_DIR, t.format(session_id=session_id)) for t in [IMU_FILE_TEMPLATE, EMG_FILE_TEMPLATE, GROUND_TRUTH_FILE_TEMPLATE]]
        if not all(os.path.exists(p) for p in paths):
            print(f"警告: 会话 {session_id} 的文件不完整，已跳过。")
            continue
        print("  1. 正在加载和解析数据...")
        df_imu = parse_imu_data(paths[0])
        df_emg = parse_emg_data(paths[1])
        df_gt = parse_ground_truth(paths[2], ACTION_LABELS)
        if any(df.empty for df in [df_imu, df_emg, df_gt]):
            print(f"警告: 会话 {session_id} 的某个数据文件为空或解析失败，已跳过。")
            continue
        
        print("  2. 正在计算关节角度和角速度...")
        df_kinematics = calculate_joint_kinematics(df_imu)
        
        print("  3. 正在对齐和插值数据 (以EMG时间戳为基准)...")
        base_index = df_emg.index
        df_to_interpolate = pd.concat([df_imu, df_kinematics], axis=1)

        if len(df_to_interpolate.index) > 1:
            interp_func_imu_kin = interp1d(
                df_to_interpolate.index.astype(np.int64), df_to_interpolate.values, 
                axis=0, kind='linear', fill_value="extrapolate", bounds_error=False
            )
            df_imu_kin_resampled = pd.DataFrame(
                interp_func_imu_kin(base_index.astype(np.int64)), 
                index=base_index, columns=df_to_interpolate.columns
            )
            df_merged = pd.concat([df_emg, df_imu_kin_resampled], axis=1)
        else:
            print(f"警告: 会话 {session_id} 的IMU数据点不足，无法进行插值，已跳过。")
            continue
        
        print("  4. 正在添加动作标签...")
        df_merged['label'] = -1
        for _, row in df_gt.iterrows():
            df_merged.loc[(df_merged.index >= row['start_time']) & (df_merged.index <= row['end_time']), 'label'] = row['label']
        df_final = df_merged[df_merged['label'] != -1].copy()
        
        if df_final.empty:
            print(f"警告: 会话 {session_id} 没有有效的带标签数据，已跳过。")
            continue

        all_raw_data[session_id] = df_final.copy()
            
        if len(df_final.index) > 1:
            mean_period_s = np.mean(np.diff(df_final.index.values)) / np.timedelta64(1, 's')
            fs = 1.0 / mean_period_s if mean_period_s > 0 else 0
        else:
            fs = 0
        session_sampling_rates[session_id] = fs
        print(f"  估计采样率: {fs:.2f} Hz")
        
        print("  5. 正在进行数据归一化...")
        data_cols = [col for col in df_final.columns if col != 'label']
        df_final[data_cols] = MinMaxScaler().fit_transform(df_final[data_cols])
        all_processed_data[session_id] = df_final
        print(f"  会话 {session_id} 处理完成。")

    if not all_processed_data:
        print("\n没有成功处理任何会话，程序终止。")
        return

    full_raw_df = pd.concat(all_raw_data.values(), ignore_index=False) # 【V6 新增】
    full_df = pd.concat(all_processed_data.values(), ignore_index=False)
    first_id = next(iter(all_processed_data))
    gt_for_vis = parse_ground_truth(os.path.join(DATA_DIR, GROUND_TRUTH_FILE_TEMPLATE.format(session_id=first_id)), ACTION_LABELS)
    
    visualize_raw_time_series(full_raw_df.drop(columns=['label']), gt_for_vis, RAW_TIMESERIES_VIS_OUTPUT)
    
    # 可视化归一化后的数据
    visualize_time_series(full_df.drop(columns=['label']), gt_for_vis, TIMESERIES_VIS_OUTPUT)
    
    print("\n--- 正在保存处理后的数据到HDF5文件 ---")
    try:
        with h5py.File(PROCESSED_DATA_OUTPUT, 'w') as f:
            for sid, df in all_processed_data.items():
                g = f.create_group(sid)
                g.create_dataset('data', data=df.values)
                g.create_dataset('index', data=df.index.astype(np.int64))
                g.attrs['columns'] = df.columns.tolist()
                g.attrs['sampling_rate_hz'] = session_sampling_rates.get(sid, 0)
        print(f"所有处理后的数据已保存到: {PROCESSED_DATA_OUTPUT}")
    except Exception as e:
        print(f"保存HDF5文件时出错: {e}")

    print(f"\n--- 正在提取特征 (窗口: {WINDOW_SIZE_S}s, 步长: {WINDOW_STEP_S}s) ---")
    avg_fs = np.mean(list(session_sampling_rates.values()))
    if avg_fs > 0:
        feature_df = extract_features(full_df.drop(columns=['label_name'], errors='ignore'), 'label', avg_fs, WINDOW_SIZE_S, WINDOW_STEP_S)
        print(f"特征提取完成，共生成 {len(feature_df)} 个特征向量。")
        
        print("\n--- 正在合并特征到完整数据集 ---")
        # 1. 为特征数据创建时间戳索引
        window_length = int(WINDOW_SIZE_S * avg_fs)
        step_length = int(WINDOW_STEP_S * avg_fs)
        window_start_indices = range(0, len(full_df) - window_length, step_length)
        
        num_features = len(feature_df)
        window_start_times = [full_df.index[i] for i in window_start_indices[:num_features]]
        
        feature_df_timed = feature_df.copy()
        if len(window_start_times) == len(feature_df_timed):
            feature_df_timed.index = pd.to_datetime(window_start_times)
            feature_df_timed = feature_df_timed.rename_axis('datetime')

            # 2. 使用 merge_asof 将特征合并到原始高频数据帧
            #    对于 full_df 中的每一行，它会找到特征数据中时间戳最接近且不晚于自己的那一行
            merged_with_features_df = pd.merge_asof(
                left=full_df.sort_index(),
                right=feature_df_timed.drop(columns='label').sort_index(), # 不合并label，避免冲突
                left_index=True,
                right_index=True,
                direction='backward' # 向后查找，确保窗口内的所有点都匹配到窗口开始时的特征
            ).dropna() # 删除头部可能因没有对应特征而产生的NaN值

            # 3. 保存合并后的数据
            print(f"--- 正在保存合并后的特征数据到HDF5文件: {MERGED_FEATURE_DATA_OUTPUT} ---")
            try:
                with h5py.File(MERGED_FEATURE_DATA_OUTPUT, 'w') as f:
                    g = f.create_group('merged_data_with_features')
                    g.create_dataset('data', data=merged_with_features_df.values)
                    g.create_dataset('index', data=merged_with_features_df.index.astype(np.int64))
                    g.attrs['columns'] = merged_with_features_df.columns.tolist()
                print(f"合并后的数据已保存到: {MERGED_FEATURE_DATA_OUTPUT}")
            except Exception as e:
                print(f"保存合并的HDF5文件时出错: {e}")

            # 4. 可视化合并后的数据
            visualize_merged_data(merged_with_features_df, gt_for_vis, MERGED_FEATURE_VIS_OUTPUT)
        else:
            print("警告：特征向量数量与窗口起始时间数量不匹配，跳过特征合并步骤。")


        # 继续进行原有的特征可分离性分析
        id_to_name = {v: k for k, v in ACTION_LABELS.items()}
        analyze_and_visualize_features(feature_df, id_to_name, SEPARABILITY_REPORT_OUTPUT, FEATURE_VIS_OUTPUT)
    else:
        print("平均采样率为0，无法提取特征。")
    
    try:
        label_map = {'name_to_id': ACTION_LABELS, 'id_to_name': {v: k for k, v in ACTION_LABELS.items()}}
        with open(LABEL_MAP_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=4)
        print(f"\n标签映射已保存到: {LABEL_MAP_OUTPUT}")
    except Exception as e:
        print(f"保存标签映射文件时出错: {e}")
        
    print("\n--- 所有任务处理完毕 ---")

if __name__ == '__main__':
    main()