import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

# 设置 numpy 输出选项，以完整显示数组
np.set_printoptions(threshold=np.inf)

# 读取数据集
data_file = 'cleaned_data-0.csv'
try:
    data = pd.read_csv(data_file, on_bad_lines='skip', low_memory=False)
except pd.errors.ParserError as e:
    print(f"解析错误: {e}")
    print("尝试手动修复 CSV 文件或检查数据格式。")
    exit()

# 检查数据集的列名
feature_names = data.columns[:-1].tolist()  # 假设最后一列是目标变量
target_name = data.columns[-1]

# 分离特征和目标变量
X = data[feature_names]
y = data[target_name]

# 检查目标变量的分布
print("目标变量的分布:")
print(y.value_counts())

# 如果目标变量中只有一个类别，提示用户
if len(y.unique()) < 2:
    print("错误：目标变量中只有一个类别，无法进行分类任务。请检查数据集。")
    exit()

# 显示数据集的前几行
print("\n数据集的前几行:")
print(data.head())

# 检查数据类型
print("\n数据类型:")
print(data.dtypes)

# 检查是否有缺失值
print("\n缺失值统计:")
print(data.isnull().sum().to_string())

# 处理缺失值
imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
X_imputed = imputer.fit_transform(X)

# 将处理后的数据转换回 DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# 处理无穷大值
X_imputed.replace([np.inf, -np.inf], np.nan, inplace=True)

# 再次处理缺失值（处理无穷大值后可能产生的新缺失值）
X_imputed.fillna(X_imputed.mean(), inplace=True)

# 检查处理后的数据
print("\n处理后的数据前几行:")
print(X_imputed.head())

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# 检查训练集和测试集的目标变量分布
print("\n训练集目标变量的分布:")
print(y_train.value_counts())
print("\n测试集目标变量的分布:")
print(y_test.value_counts())

# 如果训练集中只有一个类别，提示用户
if len(y_train.unique()) < 2:
    print("错误：训练集中只有一个类别，无法进行分类任务。请调整数据分割参数。")
    exit()

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用 SMOTE 进行过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# 创建GBDT模型
gbdt = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 训练模型
gbdt.fit(X_train_resampled, y_train_resampled)

# 获取特征重要性
feature_importances = gbdt.feature_importances_

# 创建特征重要性表格
feature_importance_df = pd.DataFrame({
    '特征名称': feature_names,
    '贡献度': feature_importances
})

# 打印特征重要性表格
print("\n特征重要性表格:")
print(feature_importance_df.sort_values(by='贡献度', ascending=False).to_string(index=False))

# 提取贡献度最高的前20个特征
top_features = feature_importance_df.sort_values(by='贡献度', ascending=False).head(20)['特征名称'].tolist()
print("\n提取的前20个特征:")
print(top_features)

# 使用前20个特征重新训练模型
X_train_top = X_train_resampled[:, [feature_names.index(f) for f in top_features]]
X_test_top = X_test_scaled[:, [feature_names.index(f) for f in top_features]]

# 重新训练模型
gbdt_top = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbdt_top.fit(X_train_top, y_train_resampled)

# 保存模型
model_filename = 'gbdt_model_top20.pkl'
joblib.dump(gbdt_top, model_filename)
print(f"\n模型已保存到 {model_filename}")

# 进行预测
predictions = gbdt_top.predict(X_test_top)

# 查看预测概率
probabilities = gbdt_top.predict_proba(X_test_top)
print("\n预测概率:")
print(probabilities)

# 调整决策阈值
threshold = 0.3  # 可以根据实际情况调整阈值
predictions_adjusted = (probabilities[:, 1] >= threshold).astype(int)

# 输出调整后的预测结果
print("\n调整后的预测结果:")
print(predictions_adjusted)

# 计算评估指标
roc_auc = roc_auc_score(y_test, probabilities[:, 1])
precision = precision_score(y_test, predictions_adjusted)
recall = recall_score(y_test, predictions_adjusted)
conf_matrix = confusion_matrix(y_test, predictions_adjusted)

print(f"\nAUC-ROC: {roc_auc:.4f}")
print(f"Precision: {1-precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# 用户输入的特征数据示例
user_input = {}
for feature in top_features:
    user_input[feature] = float(input(f"请输入 {feature} 的值: "))

# 将用户输入的特征数据转换为 DataFrame
user_input_df = pd.DataFrame([user_input])

# 确保用户输入的特征数据包含所有必要的特征列
if not set(top_features).issubset(set(user_input_df.columns)):
    print("用户输入的特征数据缺少必要的特征列")
else:
    # 提取用户输入的特征
    user_input_top = user_input_df[top_features]

    # 使用模型进行预测
    user_prediction_prob = gbdt_top.predict_proba(user_input_top)
    print("\n用户输入数据的预测概率:")
    print(user_prediction_prob)

    # 输出用户是否被欺诈的概率
    fraud_probability = user_prediction_prob[0, 1]
    print(f"用户被欺诈的概率: {fraud_probability:.4f}")
