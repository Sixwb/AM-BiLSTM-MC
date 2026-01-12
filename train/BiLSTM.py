import pandas as pd
import numpy as np
import h5py
import json
import os
from sklearn.model_selection import train_test_split
# compute_class_weight 已不再需要
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =============================================================================
# --- 1. 配置区域 ---
# =============================================================================
# 输入数据和标签映射文件路径
PROCESSED_DATA_FILE = ''
LABEL_MAP_FILE = ''
FONT_PATH = 'ChillRoundGothic_Regular.otf'

# 模型训练参数
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 0.001
LSTM_UNITS = 128

# 输出模型路径
MODEL_SAVE_PATH = 'bilstm_model.keras'


# =============================================================================
# --- 2. 辅助函数 ---
# =============================================================================

def setup_chinese_font(font_path=None):
    try:
        if font_path and os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name(), 'sans-serif']
            print(f"成功加载字体: {font_path}")
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"警告：设置中文字体失败，图表标签可能无法正常显示。错误: {e}")

def load_processed_data(hdf5_path, label_map_path):
    """从HDF5加载特征合并数据，并从JSON加载标签映射。"""
    print(f"从以下路径加载处理好的数据: {hdf5_path}")
    processed_sessions = {}
    try:
        with h5py.File(hdf5_path, 'r') as f:
            for session_id in f.keys():
                if 'data' not in f[session_id]: continue
                group = f[session_id]
                data = group['data'][:]
                index = pd.to_datetime(group['index'][:])
                columns = [col.decode('utf-8') if isinstance(col, bytes) else col for col in group.attrs['columns']]
                df = pd.DataFrame(data, index=index, columns=columns)
                processed_sessions[session_id] = df
        print(f"数据加载完成。共加载 {len(processed_sessions)} 个会话。")
    except Exception as e:
        print(f"加载HDF5文件时出错: {e}")
        return None, None
        
    print(f"从以下路径加载标签映射: {label_map_path}")
    try:
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        print("标签映射加载完成。")
        return processed_sessions, label_map
    except Exception as e:
        print(f"加载JSON标签映射时出错: {e}")
        return processed_sessions, None

def prepare_data_for_seq2seq(processed_sessions, label_map, test_size, random_state):
    """为序列到序列任务准备数据，包括拆分会话、独热编码。"""
    if not processed_sessions or not label_map:
        return [None] * 8

    session_ids = list(processed_sessions.keys())
    
    feature_columns = [col for col in next(iter(processed_sessions.values())).columns if col != 'label']
    session_data = [df[feature_columns].values for df in processed_sessions.values()]
    session_labels_original = [df['label'].values.astype(int) for df in processed_sessions.values()]

    all_labels_flat = np.concatenate(session_labels_original)
    original_unique_labels = sorted(list(np.unique(all_labels_flat)))
    new_label_mapping = {original_id: new_id for new_id, original_id in enumerate(original_unique_labels)}
    num_classes = len(new_label_mapping)
    
    session_labels_remapped = [np.array([new_label_mapping.get(label_id, -1) for label_id in labels]) for labels in session_labels_original]
    session_labels_onehot = [tf.keras.utils.to_categorical(labels, num_classes=num_classes) for labels in session_labels_remapped]

    train_session_ids, test_session_ids = train_test_split(session_ids, test_size=test_size, random_state=random_state)
    print(f"\n总会话数: {len(session_ids)}")
    print(f"训练会话 ({len(train_session_ids)}): {train_session_ids}")
    print(f"测试会话 ({len(test_session_ids)}): {test_session_ids}")

    X_train_list = [session_data[session_ids.index(sid)] for sid in train_session_ids]
    y_train_list = [session_labels_onehot[session_ids.index(sid)] for sid in train_session_ids]
    X_test_list = [session_data[session_ids.index(sid)] for sid in test_session_ids]
    y_test_list = [session_labels_onehot[session_ids.index(sid)] for sid in test_session_ids]

    input_shape = (None, X_train_list[0].shape[1])
    print(f"\n模型的输入形状: (时间步, 特征数) = {input_shape}")
    print(f"类别数量 (重映射后): {num_classes}")

    return X_train_list, y_train_list, X_test_list, y_test_list, input_shape, num_classes, new_label_mapping

def create_dataset(X_list, y_list, batch_size):
    def generator():
        for x, y in zip(X_list, y_list):
            yield x, y

    output_signature = (
        tf.TensorSpec(shape=(None, X_list[0].shape[1]), dtype=tf.float32),
        tf.TensorSpec(shape=(None, y_list[0].shape[1]), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.shuffle(buffer_size=len(X_list))

    padded_dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None, X_list[0].shape[1]]),
            tf.TensorShape([None, y_list[0].shape[1]])
        ),
        padding_values=(0.0, 0.0)
    )
    return padded_dataset.prefetch(tf.data.AUTOTUNE)

def build_basic_bilstm_model(input_shape, num_classes, lstm_units, learning_rate):
    print("\n--- 正在构建基础 BiLSTM 模型 (单层) ---")
    model_input_shape = (input_shape[0], input_shape[1])

    model = keras.models.Sequential([
        keras.layers.Masking(mask_value=0.0, input_shape=model_input_shape),
        keras.layers.Bidirectional(keras.layers.LSTM(units=lstm_units, return_sequences=True)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def plot_training_history(history):
    pass

# =============================================================================
# --- 3. 主执行流程 ---
# =============================================================================
def main():
    print("--- 开始基础 BiLSTM 模型训练 ---")

    # 1. 加载数据
    processed_sessions, label_map = load_processed_data(PROCESSED_DATA_FILE, LABEL_MAP_FILE)
    if not processed_sessions or not label_map:
        return

    # 2. 准备序列到序列格式的数据
    X_train_list, y_train_list, X_test_list, y_test_list, input_shape, num_classes, new_label_mapping = \
        prepare_data_for_seq2seq(processed_sessions, label_map, TEST_SIZE, RANDOM_STATE)
    if not X_train_list:
        return

    # 3. 创建数据集
    train_dataset = create_dataset(X_train_list, y_train_list, BATCH_SIZE)
    test_dataset = create_dataset(X_test_list, y_test_list, BATCH_SIZE)

    # 4. 构建模型
    model = build_basic_bilstm_model(input_shape, num_classes, LSTM_UNITS, LEARNING_RATE)

    # 5. 定义回调函数
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, restore_best_weights=True
    )

    # 6. 训练模型
    print("\n--- 正在训练模型 ---")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[early_stopping_callback]
    )
    print("--- 模型训练完成 ---")

    # 7. 评估和可视化
    print("\n--- 在测试集上评估最终模型 ---")
    loss, accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"最终测试集损失: {loss:.4f}")
    print(f"最终测试集准确率: {accuracy:.4f}")

    # 8. 保存模型
    print("\n--- 保存模型 ---")
    model.save(MODEL_SAVE_PATH)
    print(f"模型成功保存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()