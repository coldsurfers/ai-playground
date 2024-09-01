# 필요한 라이브러리 불러오기
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# 붓꽃 데이터셋 불러오기
iris = datasets.load_iris()
# X = iris.data
X = iris.data[:, :2]  # Use only the first two features for visualization
y = iris.target
target_names = iris.target_names

# 데이터셋을 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 데이터 표준화 (평균이 0, 분산이 1이 되도록 변환)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 퍼셉트론 모델 생성 및 학습
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=42)
ppn.fit(X_train, y_train)

# 예측
y_pred = ppn.predict(X_test)

# 정확도 평가
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')

# 분류 보고서 출력 (정밀도, 재현율, F1 점수 등)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Plotting only the first two features
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Iris Dataset')
plt.legend()
plt.show()

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the mesh grid
Z = ppn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary of Perceptron Model')
plt.show()


# Generate learning curves
train_sizes, train_scores, test_scores = learning_curve(
    ppn, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Plot learning curves
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation score')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curves (Perceptron)')
plt.legend(loc='best')
plt.show()