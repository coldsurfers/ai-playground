import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 붓꽃 데이터셋 불러오기
iris = load_iris()
X = iris.data  # 입력 데이터 (특성)
y = iris.target  # 레이블 (클래스)

# 레이블을 원-핫 인코딩
encoder = OneHotEncoder(sparse_output=False)  # sklearn 버전이 1.2 이상일 경우 sparse_output 사용
y = encoder.fit_transform(y.reshape(-1, 1))

# 훈련 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 활성화 함수 및 그 미분 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 안정적인 softmax 계산을 위한 정규화
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 모델 초기화
input_size = X_train.shape[1]   # 입력 데이터의 특징 수 (4)
hidden_size = 10  # 은닉층 노드 수
output_size = y_train.shape[1]  # 출력 노드 수 (3개의 클래스)

# 가중치 초기화
np.random.seed(42)  # 결과를 재현 가능하게 하기 위해 난수 시드 설정
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# 하이퍼파라미터
learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    # 순방향 전파
    Z1 = np.dot(X_train, W1) + b1  # 은닉층으로의 입력
    A1 = sigmoid(Z1)  # 은닉층 활성화
    Z2 = np.dot(A1, W2) + b2  # 출력층으로의 입력
    A2 = softmax(Z2)  # 출력층 활성화 (Softmax)

    # 손실 계산 (교차 엔트로피 손실 함수)
    loss = -np.mean(np.sum(y_train * np.log(A2 + 1e-9), axis=1))  # 안정성 향상을 위한 작은 값 추가

    # 역전파
    dZ2 = A2 - y_train  # 출력층에서의 오차
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
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
