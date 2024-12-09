import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

# 读取训练集和测试集
train_data = pd.read_excel('train_set.xlsx')
test_data = pd.read_excel('test_set.xlsx')

# 分离特征和目标变量
feature_names = train_data.columns[:-1].tolist()  # 假设最后一列是目标变量
target_name = train_data.columns[-1]
X_train = train_data.drop(columns=[target_name])
y_train = train_data[target_name]
X_test = test_data.drop(columns=[target_name])
y_test = test_data[target_name]

# 标准化特征
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 使用随机森林评估特征重要性，选择前20个特征
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
feature_importances = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 获取前20个重要特征
top_features = feature_importances.head(20)['Feature'].tolist()
print("前20个重要特征:")
print(top_features)

# 使用前20个特征进行训练和测试
X_train_top = X_train_scaled[top_features]
X_test_top = X_test_scaled[top_features]
X_train_top_scaled = pd.DataFrame(scaler.fit_transform(X_train_top), columns=X_train_top.columns)
X_test_top_scaled = pd.DataFrame(scaler.transform(X_test_scaled[top_features]), columns=X_test_top.columns)
# 构建深度学习模型
model = Sequential([
    Dense(32, input_dim=len(top_features), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 计算类别权重
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# 添加早停机制
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 进行训练
model.fit(X_train_top, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping], class_weight=class_weights)

# 预测概率和结果
y_pred_prob = model.predict(X_test_top).flatten()  # 预测概率值

# 打印测试集中前几个样本的预测概率
num_samples_to_print = 10  # 你希望打印的样本数量
print(f"\n前{num_samples_to_print}个测试样本的预测概率:")
for i in range(min(num_samples_to_print, len(y_pred_prob))):
    print(f"样本 {i+1}: {y_pred_prob[i]:.4f}")

y_pred = (y_pred_prob >= 0.7).astype(int)  # 二分类预测
print(y_pred)
# 评估模型
auc = roc_auc_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_text = classification_report(y_test, y_pred)

print("\n模型评估指标:")
print(f"AUC-ROC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print("\nClassification Report:")
print(classification_report_text)

# 用户输入预测
while True:
    input_values = input(f"请按顺序输入以下特征值，用空格分隔：\n{' '.join(top_features)}\n")
    input_list = input_values.split()
    print("用户输入：", input_list)  # 打印用户输入
    if len(input_list) == len(top_features):
        try:
            user_input = dict(zip(top_features, map(float, input_list)))  # 转换为字典
            user_input_df = pd.DataFrame([user_input]).reindex(columns=top_features)
            # 标准化用户输入的数据
            input_data_scaled = scaler.transform(user_input_df)

            # 使用模型进行预测
            user_prediction_prob = model.predict(input_data_scaled).flatten()
            fraud_probability = user_prediction_prob[0]  # 获得欺诈的预测概率

            # 输出预测结果
            print("\n用户输入数据的预测结果:")
            fraud_probability_percentage = fraud_probability * 100  # 转换为百分比
            print(f"这次预测为欺诈的概率: {fraud_probability_percentage:.2f}%")
            break  # 退出循环

        except ValueError as e:
            print(f"转换错误: {e}. 确保所有特征值为数字。")
    else:
        print(f"输入的特征数量不对，应为{len(top_features)}个，请重新输入。")