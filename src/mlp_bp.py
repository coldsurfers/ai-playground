import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 붓꽃 데이터셋 불러오기
iris = load_iris()
X = iris.data  # 입력 데이터 (특성)
y = iris.target  # 레이블 (클래스)

# 레이블을 원-핫 인코딩
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# 훈련 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 활성화 함수 및 그 미분 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 모델 초기화
input_size = X_train.shape[1]
hidden_size = 10  # 은닉층 노드 수
output_size = y_train.shape[1]

# 가중치 초기화
W1 = np.random.rand(input_size, hidden_size)
b1 = np.random.rand(hidden_size)
W2 = np.random.rand(hidden_size, output_size)
b2 = np.random.rand(output_size)

# 하이퍼파라미터
learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    # 순방향 전파
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # 손실 계산 (교차 엔트로피 손실을 사용하지 않고 간단히 MSE 사용)
    loss = np.mean((A2 - y_train) ** 2)

    # 역전파
    dA2 = A2 - y_train
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0)

    # 가중치 및 바이어스 업데이트
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # 1000 에포크마다 손실 출력
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss}')

# 테스트 데이터에 대한 예측
Z1_test = np.dot(X_test, W1) + b1
A1_test = sigmoid(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = sigmoid(Z2_test)

# 정확도 계산
predictions = np.argmax(A2_test, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions == y_true)
print(f'Accuracy: {accuracy * 100:.2f}%')
