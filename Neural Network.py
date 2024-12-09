import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import joblib

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

# 输出训练集和测试集的目标变量分布
print("训练集目标变量的分布:")
print(y_train.value_counts())

print("\n测试集目标变量的分布:")
print(y_test.value_counts())
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

# 打印前20个重要特征
print("\n提取的前20个特征变量:")
for idx, feature in enumerate(top_features, 1):
    print(f"{idx}. {feature}")

# 使用前20个特征进行训练和测试
X_train_top = X_train_scaled[top_features]
X_test_top = X_test_scaled[top_features]
X_train_top_scaled = pd.DataFrame(scaler.fit_transform(X_train_top), columns=X_train_top.columns)
X_test_top_scaled = pd.DataFrame(scaler.transform(X_test_scaled[top_features]), columns=X_test_top.columns)

# 计算类别权重
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# 构建模型组合
gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, scale_pos_weight=class_weights[1])
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# 集成模型 (投票分类器)
voting_model = VotingClassifier(
    estimators=[
        ('gbdt', gbdt),
        ('xgb', xgb),
        ('rf', rf)
    ],
    voting='soft'  # 使用概率投票
)

# 训练模型
voting_model.fit(X_train_top, y_train)

# 保存模型
model_filename = 'voting_model.pkl'
joblib.dump(voting_model, model_filename)
print(f"\n模型已保存到 {model_filename}")

# 评估模型
y_pred_prob = voting_model.predict_proba(X_test_top)[:, 1]
y_pred = (y_pred_prob >= 0.2).astype(int)  # 使用 0.2 作为阈值

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
    if len(input_list) == len(top_features):
        try:
            user_input = dict(zip(top_features, map(float, input_list)))  # 转换为字典
            user_input_df = pd.DataFrame([user_input]).reindex(columns=top_features)

            # 标准化用户输入的数据
            input_data_scaled = scaler.transform(user_input_df)

            # 使用模型进行预测
            user_prediction_prob = voting_model.predict_proba(input_data_scaled)[:, 1]
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
