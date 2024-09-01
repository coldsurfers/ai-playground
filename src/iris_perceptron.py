# 필요한 라이브러리 불러오기
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# 붓꽃 데이터셋 불러오기
iris = datasets.load_iris()
X = iris.data
y = iris.target

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