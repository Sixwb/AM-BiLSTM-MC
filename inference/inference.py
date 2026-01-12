import pandas as pd
import numpy as np
import h5py
import json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 配置 ---
MODEL_PATH = 'bilstm_model.keras' # 更新模型路径
LABEL_MAP_FILE = '' # 更新标签映射路径
PROCESSED_DATA_FOR_PARAMS = '' # 用于加载参数的HDF5文件

# 新的原始数据文件路径 (示例)
NEW_IMU_FILE = ''
NEW_EMG_FILE = ''
NEW_GT_FILE = ''   #

PREDICTION_OUTPUT_FILE = 'bo_bilstm_activity_recognition_results_v1.csv'
CONFUSION_MATRIX_OUTPUT_FILE = 'bo_bilstm_inference_confusion_matrix_v1.png'
LABEL_COMPARISON_PLOT_FILE = 'bo_bilstm_inference_label_comparison_v1.png'

def parse_time_string_to_datetime(time_str):
    try: dt_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    except ValueError:
        parts = time_str.split('.')
        if len(parts) == 2:
            ms = parts[1]; ms = ms[:6] if len(ms) > 6 else ms.ljust(6, '0')
            try: dt_obj = datetime.strptime(f"{parts[0]}.{ms}", '%H:%M:%S.%f')
            except ValueError:
                try: dt_obj = datetime.strptime(parts[0], '%H:%M:%S')
                except ValueError: return None
        else:
            try: dt_obj = datetime.strptime(time_str, '%H:%M:%S')
            except ValueError: return None
    try: return dt_obj.replace(year=1970, month=1, day=1)
    except AttributeError: return None

def parse_time_relative_to_base(time_str, base_time_dt_obj):
    """计算时间字符串相对于基准时间的秒数"""
    dt_obj = parse_time_string_to_datetime(time_str)
    if dt_obj is None or base_time_dt_obj is None: return None
    time_delta = dt_obj - base_time_dt_obj
    if -timedelta(days=1).total_seconds() < time_delta.total_seconds() < 0:
        time_delta += timedelta(days=1)
    elif time_delta.total_seconds() < -timedelta(days=1).total_seconds(): return None
    return time_delta.total_seconds()

def load_sensor_data_inference(filepath, sensor_type, global_base_time):
    print(f"正在加载 {sensor_type} 数据用于推理: {os.path.basename(filepath)}")
    data = []
    if sensor_type == 'IMU':
        pattern = re.compile(
            r"IMU1_.*?AX:\s*(-?[\d\.E\+\-]+)\s*AY:\s*(-?[\d\.E\+\-]+)\s*AZ:\s*(-?[\d\.E\+\-]+).*?"
            r"IMU2_.*?AX:\s*(-?[\d\.E\+\-]+)\s*AY:\s*(-?[\d\.E\+\-]+)\s*AZ:\s*(-?[\d\.E\+\-]+).*?"
            r"IMU3_.*?AX:\s*(-?[\d\.E\+\-]+)\s*AY:\s*(-?[\d\.E\+\-]+)\s*AZ:\s*(-?[\d\.E\+\-]+).*?"
            r"time:\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
        )
        column_names = ['IMU1_AX', 'IMU1_AY', 'IMU1_AZ', 'IMU2_AX', 'IMU2_AY', 'IMU2_AZ', 'IMU3_AX', 'IMU3_AY', 'IMU3_AZ']
        value_indices = list(range(1, 10)); time_index = 10
    elif sensor_type == 'EMG':
        pattern = re.compile(
            r"EMG_1:\s*(-?[\d\.E\+\-]+)\s*EMG_2:\s*(-?[\d\.E\+\-]+)\s*EMG_3:\s*(-?[\d\.E\+\-]+)\s*"
            r"EMG_4:\s*(-?[\d\.E\+\-]+)\s*EMG_5:\s*(-?[\d\.E\+\-]+)\s*EMG_6:\s*(-?[\d\.E\+\-]+)\s*"
            r"time:\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
        )
        column_names = [f'EMG_{i}' for i in range(1, 7)]
        value_indices = list(range(1, 7)); time_index = 7
    else: return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip(); match = pattern.search(line)
                if match:
                    time_seconds = parse_time_relative_to_base(match.group(time_index), global_base_time)
                    if time_seconds is None: continue
                    try:
                        row_values = [float(match.group(j)) for j in value_indices]
                        row = dict(zip(column_names, row_values)); row['time_seconds'] = time_seconds
                        data.append(row)
                    except ValueError: continue
    except Exception: return None
    if not data: return None
    df = pd.DataFrame(data).sort_values(by='time_seconds').drop_duplicates(subset='time_seconds', keep='first').set_index('time_seconds')
    df.dropna(axis=0, how='any', inplace=True)
    print(f"加载完成 {os.path.basename(filepath)}。形状: {df.shape}")
    return df


def align_data_inference(imu_df, emg_df):
    print("正在对齐 IMU 和 EMG 数据用于推理...")
    if imu_df is None or emg_df is None or imu_df.empty or emg_df.empty: return None
    if len(imu_df) < 2 or len(emg_df) < 2: return None
    min_time = max(imu_df.index.min(), emg_df.index.min())
    max_time = min(imu_df.index.max(), emg_df.index.max())
    if min_time >= max_time: return None
    target_time_index = imu_df.loc[min_time:max_time].index.sort_values()
    if target_time_index.empty or len(target_time_index) < 2: return None
    aligned_emg_data = {}
    emg_cols = [col for col in emg_df.columns if col.startswith('EMG')]
    if not emg_cols: return None
    emg_for_interp = emg_df.loc[emg_df.index.min():emg_df.index.max()].sort_index()
    if len(emg_for_interp) < 2: return None
    for col in emg_cols:
        try:
            interp_func = interp1d(emg_for_interp.index, emg_for_interp[col], kind='linear', bounds_error=False, fill_value=np.nan)
            aligned_emg_data[col] = interp_func(target_time_index)
        except Exception: return None
    aligned_emg_df = pd.DataFrame(aligned_emg_data, index=target_time_index)
    combined_df = pd.concat([imu_df.loc[target_time_index].sort_index(), aligned_emg_df], axis=1)
    if combined_df.isnull().any(axis=1).sum() > 0:
        combined_df = combined_df.dropna()
        if combined_df.empty: return None
    print(f"数据对齐完成。合并后的数据形状: {combined_df.shape}")
    return combined_df


def apply_filtering_inference(df, filter_order, cutoff_frequency_hz, sampling_rate_hz):
    if df is None or df.empty: return df
    if sampling_rate_hz is None or sampling_rate_hz <= 0: return df
    nyquist = 0.5 * sampling_rate_hz
    normal_cutoff = cutoff_frequency_hz / nyquist
    if normal_cutoff >= 1.0 or normal_cutoff <= 0: return df
    print(f"正在应用低通滤波 (阶数: {filter_order}, 采样率: {sampling_rate_hz:.2f} Hz, 截止: {cutoff_frequency_hz} Hz)...")
    try: b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
    except ValueError: return df
    filtered_df = df.copy()
    for col in filtered_df.columns:
        if pd.api.types.is_numeric_dtype(filtered_df[col]):
            try:
                min_len = max(3 * (len(a) - 1), 3 * (len(b) - 1))
                if len(filtered_df[col].dropna()) >= min_len:
                    valid_idx = filtered_df[col].notna()
                    filtered_df.loc[valid_idx, col] = filtfilt(b, a, filtered_df.loc[valid_idx, col])
            except Exception: filtered_df[col] = df[col] # 出错则保留原始
    print("滤波应用完成。")
    return filtered_df

def calculate_central_moments_inference(df, order=2, window_size=10):
    print(f"正在为推理数据计算中心矩特征 (阶数: {order}, 窗口: {window_size})...")
    if df is None or df.empty: return df
    feature_df = pd.DataFrame(index=df.index)
    original_cols = df.columns.tolist()
    for col in original_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            feature_df[col] = df[col]; continue
        feature_df[col] = df[col] # 原始列
        if order >= 1: feature_df[f"{col}_mean_w{window_size}"] = df[col].rolling(window=window_size, min_periods=1).mean()
        if order >= 2: feature_df[f"{col}_std_w{window_size}"] = df[col].rolling(window=window_size, min_periods=1).std()
        if order >= 3: feature_df[f"{col}_skew_w{window_size}"] = df[col].rolling(window=window_size, min_periods=max(1,window_size//2)).skew()
        if order >= 4: feature_df[f"{col}_kurt_w{window_size}"] = df[col].rolling(window=window_size, min_periods=max(1,window_size//2)).kurt()
    feature_df.fillna(method='bfill', inplace=True)
    feature_df.fillna(method='ffill', inplace=True)
    feature_df.fillna(0, inplace=True)
    print(f"推理数据中心矩特征计算完成。形状: {feature_df.shape}")
    return feature_df


def load_inference_parameters(hdf5_path):

    print(f"正在从 {hdf5_path} 加载推理参数...")
    params = {'global_base_time_dt': None, 'filter_order': None, 'cutoff_frequency_hz': None,
              'sampling_rate_to_use': None, 'central_moment_order': None, 'moment_window_size': None}
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'global_base_time_iso' in f.attrs:
                try:
                    base_time_iso = f.attrs['global_base_time_iso']
                    params['global_base_time_dt'] = datetime.fromisoformat(base_time_iso).replace(year=1970, month=1, day=1)
                except Exception: pass
            params['filter_order'] = f.attrs.get('filter_order')
            params['cutoff_frequency_hz'] = f.attrs.get('cutoff_frequency_hz')
            params['central_moment_order'] = f.attrs.get('central_moment_order', 2)
            params['moment_window_size'] = f.attrs.get('moment_window_size', 10)

            if 'manual_sampling_rate_hz' in f.attrs and f.attrs['manual_sampling_rate_hz'] > 0:
                params['sampling_rate_to_use'] = float(f.attrs['manual_sampling_rate_hz'])
            else: # 尝试从第一个会话组获取 (如果存在)
                for key in f.keys():
                    if isinstance(f[key], h5py.Group) and 'sampling_rate_hz' in f[key].attrs:
                        sr = f[key].attrs['sampling_rate_hz']
                        if sr is not None and sr > 0: params['sampling_rate_to_use'] = float(sr); break
        missing = [k for k,v in params.items() if v is None and k not in ['central_moment_order', 'moment_window_size']] # 后两者有默认值
        if missing: print(f"警告: 未能加载所有必要参数: {', '.join(missing)}。")
        else: print("推理参数加载成功。")
    except Exception as e: print(f"加载参数时出错: {e}")
    return params


def load_model_and_label_map(model_path, label_map_path):
    model, original_name_to_id, remapped_id_to_name = None, None, None
    try: model = tf.keras.models.load_model(model_path); print("模型加载成功。")
    except Exception as e: print(f"加载模型失败: {e}")
    try:
        with open(label_map_path, 'r', encoding='utf-8') as f: label_data = json.load(f)
        original_name_to_id = label_data.get('name_to_id')
        if 'remapped_id_to_name' in label_data:
            remapped_id_to_name = {int(k): v for k, v in label_data['remapped_id_to_name'].items()}
        if original_name_to_id and remapped_id_to_name: print("标签映射加载成功。")
        else: print("警告: 标签映射不完整。")
    except Exception as e: print(f"加载标签映射失败: {e}")
    return model, original_name_to_id, remapped_id_to_name


def preprocess_new_data(imu_file, emg_file, params):
    print("\n--- 正在预处理新的原始数据 ---")
    if any(params[k] is None for k in ['global_base_time_dt', 'filter_order', 'cutoff_frequency_hz', 'sampling_rate_to_use', 'central_moment_order', 'moment_window_size']):
        print("错误: 预处理参数不完整。")
        return None

    imu_df = load_sensor_data_inference(imu_file, 'IMU', params['global_base_time_dt'])
    emg_df = load_sensor_data_inference(emg_file, 'EMG', params['global_base_time_dt'])
    if imu_df is None or emg_df is None: return None

    aligned_df = align_data_inference(imu_df, emg_df)
    if aligned_df is None: return None

    filtered_df = apply_filtering_inference(aligned_df, params['filter_order'], params['cutoff_frequency_hz'], params['sampling_rate_to_use'])
    if filtered_df is None : return None

    feature_df = calculate_central_moments_inference(filtered_df, order=params['central_moment_order'], window_size=params['moment_window_size'])
    print("--- 新数据预处理完成 ---")
    return feature_df


def predict_activity_sequence(model, preprocessed_df):
    print("\n--- 正在进行模型预测 ---")
    if model is None or preprocessed_df is None or preprocessed_df.empty: return None
    input_data = preprocessed_df.values
    input_data_batch = np.expand_dims(input_data, axis=0)
    print(f"输入模型的数据形状: {input_data_batch.shape}")
    try:
        predictions_onehot = model.predict(input_data_batch, verbose=0)
        predicted_labels_remapped_ids = np.argmax(np.squeeze(predictions_onehot, axis=0), axis=-1)
        print("模型预测完成。")
        return predicted_labels_remapped_ids
    except Exception as e:
        print(f"模型预测时发生错误: {e}")
        try: print(f"模型期望输入: {model.input_shape}")
        except: pass
        return None


def postprocess_predictions(timestamps, predicted_labels_remapped_ids, remapped_id_to_name_map):
    print("\n--- 正在后处理预测结果 ---")
    if timestamps is None or predicted_labels_remapped_ids is None or len(timestamps) != len(predicted_labels_remapped_ids) or not remapped_id_to_name_map:
        return []
    activities = []
    start_time, current_label_id = None, None
    for i, ts in enumerate(timestamps):
        label_id = predicted_labels_remapped_ids[i]
        if current_label_id is None:
            start_time, current_label_id = ts, label_id
        elif label_id != current_label_id:
            activities.append({
                'start_time_sec': start_time, 'end_time_sec': timestamps[i-1],
                'duration_sec': timestamps[i-1] - start_time,
                'predicted_label_id': int(current_label_id),
                'activity_name': remapped_id_to_name_map.get(int(current_label_id), f"未知ID_{current_label_id}")
            })
            start_time, current_label_id = ts, label_id
    if current_label_id is not None: # Add last segment
        activities.append({
            'start_time_sec': start_time, 'end_time_sec': timestamps[-1],
            'duration_sec': timestamps[-1] - start_time,
            'predicted_label_id': int(current_label_id),
            'activity_name': remapped_id_to_name_map.get(int(current_label_id), f"未知ID_{current_label_id}")
        })
    print(f"后处理完成，识别出 {len(activities)} 个动作段。")
    return activities


def save_and_display_results(activities, output_filepath, global_base_time_dt):
    if not activities: print("没有识别结果可保存/显示。"); return
    df_out = pd.DataFrame(activities)
    time_format = '%H:%M:%S.%f'
    if global_base_time_dt:
        df_out['start_time_abs'] = df_out['start_time_sec'].apply(lambda x: (global_base_time_dt + timedelta(seconds=x)).strftime(time_format))
        df_out['end_time_abs'] = df_out['end_time_sec'].apply(lambda x: (global_base_time_dt + timedelta(seconds=x)).strftime(time_format))
        cols = ['start_time_abs', 'end_time_abs', 'duration_sec', 'activity_name', 'predicted_label_id', 'start_time_sec', 'end_time_sec']
    else:
        cols = ['start_time_sec', 'end_time_sec', 'duration_sec', 'activity_name', 'predicted_label_id']

    df_out = df_out[[c for c in cols if c in df_out.columns]] #确保列存在
    try: df_out.to_csv(output_filepath, index=False, encoding='utf-8-sig'); print(f"结果已保存到: {output_filepath}")
    except Exception as e: print(f"保存结果失败: {e}")
    print("\n--- 识别出的动作段 ---")
    for _, row in df_out.iterrows():
        start_disp = row.get('start_time_abs', f"{row['start_time_sec']:.3f}s")
        end_disp = row.get('end_time_abs', f"{row['end_time_sec']:.3f}s")
        print(f"  开始: {start_disp}, 结束: {end_disp}, 持续: {row['duration_sec']:.3f}s, 动作: {row['activity_name']} (ID: {row['predicted_label_id']})")


def load_ground_truth_for_validation(filepath, original_name_to_id, global_base_time_dt):
    print(f"加载地面真实用于验证: {os.path.basename(filepath)}")
    intervals = []
    if not original_name_to_id: return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) !=3: parts = [p.strip() for p in line.strip().split('\t')]
                if len(parts) == 3:
                    s_sec = parse_time_relative_to_base(parts[0], global_base_time_dt)
                    e_sec = parse_time_relative_to_base(parts[1], global_base_time_dt)
                    act_id = original_name_to_id.get(parts[2])
                    if s_sec is not None and e_sec is not None and act_id is not None:
                        intervals.append({'start_time_sec': s_sec, 'end_time_sec': e_sec, 'action_label_id': act_id, 'action_name':parts[2]})
    except Exception: return None
    print(f"地面真实加载完成，找到 {len(intervals)} 个区间。")
    return sorted(intervals, key=lambda x: x['start_time_sec'])


def get_ground_truth_labels_for_timestamps(timestamps, gt_intervals, original_name_to_id):
    if timestamps is None or len(timestamps) == 0 or not original_name_to_id: return np.array([])
    default_id = original_name_to_id.get("过渡/未知", -1)
    true_labels = np.full(len(timestamps), default_id, dtype=int)
    if not gt_intervals: return true_labels
    print("正在为时间戳分配地面真实标签...")
    idx = 0
    for i, ts in enumerate(timestamps):
        while idx < len(gt_intervals) and gt_intervals[idx]['end_time_sec'] < ts: idx += 1
        for j in range(idx, len(gt_intervals)):
            interval = gt_intervals[j]
            if interval['start_time_sec'] <= ts <= interval['end_time_sec']:
                true_labels[i] = interval['action_label_id']; break
            if interval['start_time_sec'] > ts: break
    print("地面真实标签分配完成。")
    return true_labels

def create_label_mappings_for_eval(original_name_to_id, remapped_id_to_name):
    if not original_name_to_id or not remapped_id_to_name: return None, None
    remap_to_orig = {}
    orig_to_name = {v:k for k,v in original_name_to_id.items()}
    for remap_id, name in remapped_id_to_name.items():
        if name in original_name_to_id:
            remap_to_orig[int(remap_id)] = original_name_to_id[name]
    valid_orig_ids = sorted(list(set(remap_to_orig.values())))
    target_names = [orig_to_name.get(oid, f"ID_{oid}") for oid in valid_orig_ids]
    return remap_to_orig, target_names


def plot_confusion_matrix_eval(y_true, y_pred, labels_for_cm, class_names_for_cm, output_filename):
    if len(y_true) == 0: return
    try: plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']; plt.rcParams['axes.unicode_minus'] = False
    except: pass
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_cm)
    plt.figure(figsize=(max(8,len(class_names_for_cm)), max(6,len(class_names_for_cm)*0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_for_cm, yticklabels=class_names_for_cm)
    plt.xlabel('预测标签'); plt.ylabel('真实标签'); plt.title('混淆矩阵')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    try: plt.savefig(output_filename, dpi=150); print(f"混淆矩阵已保存: {output_filename}")
    except: pass; plt.show()


def plot_labels_over_time_eval(timestamps, y_true_orig, y_pred_orig, orig_id_to_name, accuracy, output_filename):
    if timestamps is None or len(timestamps) == 0: return
    try: plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']; plt.rcParams['axes.unicode_minus'] = False
    except: pass

    all_involved_ids = sorted(list(set(y_true_orig) | set(y_pred_orig)))
    y_ticks_labels = [orig_id_to_name.get(int(id_val), f"未知ID_{id_val}") for id_val in all_involved_ids]
    id_to_yvalue = {id_val: i for i, id_val in enumerate(all_involved_ids)}
    default_y_val = -1

    y_true_numeric = [id_to_yvalue.get(int(id_val), default_y_val) for id_val in y_true_orig]
    y_pred_numeric = [id_to_yvalue.get(int(id_val), default_y_val) for id_val in y_pred_orig]

    plt.figure(figsize=(15, max(5, len(y_ticks_labels) * 0.5)))
    plt.plot(timestamps, y_true_numeric, label='真实标签', linestyle='--', alpha=0.7, drawstyle='steps-post')
    plt.plot(timestamps, y_pred_numeric, label='预测标签', linestyle='-', alpha=0.7, drawstyle='steps-post')
    plt.yticks(ticks=range(len(y_ticks_labels)), labels=y_ticks_labels)
    plt.xlabel('时间 (秒)'); plt.ylabel('动作标签')
    plt.title(f'预测 vs 真实标签 (准确率: {accuracy:.4f})'); plt.legend(); plt.grid(True, axis='y', linestyle=':')
    plt.ylim(-0.5, len(y_ticks_labels) -0.5); plt.tight_layout()
    try: plt.savefig(output_filename, dpi=150); print(f"标签对比图已保存: {output_filename}")
    except: pass; plt.show()


if __name__ == "__main__":
    print("--- 开始BO-BiLSTM模型推理与验证 ---")
    params = load_inference_parameters(PROCESSED_DATA_FOR_PARAMS)
    if params['global_base_time_dt'] is None : exit()

    model, original_name_to_id, remapped_id_to_name = load_model_and_label_map(MODEL_PATH, LABEL_MAP_FILE)
    if model is None or original_name_to_id is None or remapped_id_to_name is None: exit()
    original_id_to_name = {v: k for k, v in original_name_to_id.items()}

    preprocessed_feature_df = preprocess_new_data(NEW_IMU_FILE, NEW_EMG_FILE, params)
    if preprocessed_feature_df is None or preprocessed_feature_df.empty: exit()
    preprocessed_timestamps = preprocessed_feature_df.index.values

    predicted_labels_remapped = predict_activity_sequence(model, preprocessed_feature_df)
    if predicted_labels_remapped is None: exit()

    recognized_acts = postprocess_predictions(preprocessed_timestamps, predicted_labels_remapped, remapped_id_to_name)
    save_and_display_results(recognized_acts, PREDICTION_OUTPUT_FILE, params['global_base_time_dt'])

    print("\n--- 开始推理结果验证 ---")
    if not os.path.exists(NEW_GT_FILE):
        print(f"错误: 地面真实文件未找到: {NEW_GT_FILE}。跳过验证。")
    else:
        gt_intervals_val = load_ground_truth_for_validation(NEW_GT_FILE, original_name_to_id, params['global_base_time_dt'])
        if gt_intervals_val:
            true_labels_orig_ids_val = get_ground_truth_labels_for_timestamps(preprocessed_timestamps, gt_intervals_val, original_name_to_id)
            if len(true_labels_orig_ids_val) == len(predicted_labels_remapped):
                remap_to_orig_map, _ = create_label_mappings_for_eval(original_name_to_id, remapped_id_to_name)
                if remap_to_orig_map:
                    pred_labels_orig_ids_val = np.array([remap_to_orig_map.get(int(rid), -1) for rid in predicted_labels_remapped])

                    valid_indices_eval = (pred_labels_orig_ids_val != -1)
                    transition_id = original_name_to_id.get("过渡/未知", -999)
                    valid_indices_eval &= (true_labels_orig_ids_val != transition_id)
                    valid_indices_eval &= (true_labels_orig_ids_val != -1)

                    y_true_final = true_labels_orig_ids_val[valid_indices_eval]
                    y_pred_final = pred_labels_orig_ids_val[valid_indices_eval]

                    if len(y_true_final) > 0:
                        acc = accuracy_score(y_true_final, y_pred_final)
                        print(f"\n评估准确率 (排除过渡/未知和无效预测): {acc:.4f}")

                        unique_eval_ids = sorted(list(set(y_true_final) | set(y_pred_final)))
                        class_names_eval_final = [original_id_to_name.get(int(l_id), f"ID_{l_id}") for l_id in unique_eval_ids]

                        print("\n分类报告:")
                        print(classification_report(y_true_final, y_pred_final, labels=unique_eval_ids, target_names=class_names_eval_final, zero_division=0))

                        plot_confusion_matrix_eval(y_true_final, y_pred_final, unique_eval_ids, class_names_eval_final, CONFUSION_MATRIX_OUTPUT_FILE)
                        plot_labels_over_time_eval(preprocessed_timestamps, true_labels_orig_ids_val, pred_labels_orig_ids_val, original_id_to_name, acc, LABEL_COMPARISON_PLOT_FILE)
                    else:
                        print("警告：过滤后没有有效数据用于评估。")
    print("\n--- 推理与验证完成 ---")