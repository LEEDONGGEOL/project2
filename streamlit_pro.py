import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
# 음수 기호가 깨지는 것을 방지
plt.rcParams['axes.unicode_minus'] = False


st.header('국건영 우울증 예측모델')

st.subheader('1.데이터 불러오기')
# 1. 데이터 불러오기 (청크 사용)
chunksize = 30000 
train_final = pd.read_csv('train_final.csv', chunksize=chunksize)

# 첫 번째 청크만 사용하여 train_final에 저장
train_final = next(train_final)
st.write(train_final.head())  # 데이터의 첫 5개 행을 출력

st.write(train_final.shape)

st.subheader('2.데이터 탐색 및 전처리')
# 2. 1인가구 여부 전처리
train_final["1인가구 여부"] = np.nan
train_final.loc[train_final["가구원수"] == 1, "1인가구 여부"] = 1
train_final["1인가구 여부"] = train_final["1인가구 여부"].fillna(0)

# '1인가구 여부' 값들의 분포 출력
st.write(train_final["1인가구 여부"].value_counts())
st.write(train_final.shape)
st.write('''
- 우울증 예측 모델에서 **'1인가구 여부'**는 중요한 피처가 될 수 있습니다.
- 1인가구는 **사회적 고립**과 **정서적 지원 부족**으로 인해 우울증 위험이 높을 수 있습니다.
- 따라서, 가구원 수가 1인 경우 '1인가구 여부'를 1로 설정하고, 그 외에는 0으로 설정하여 데이터 전처리를 진행하였습니다.''')


# 1년간 체중 증감 열의 음수 값을 절대값으로 변환 (체중 감소 -> 절대값으로 변환)
train_final["1년간 체중 증감"] = train_final["1년간 체중 증감"].apply(lambda x: abs(x) if x < 0 else x)

# 1년간 체중 증감 값들의 빈도 확인
st.write(train_final["1년간 체중 증감"].value_counts())
# 전처리 이유 설명
st.write('''
- 체중 변화는 우울증과 같은 정신 건강 상태와 밀접한 관련이 있을 수 있습니다.
- 따라서 체중이 증가했는지 또는 감소했는지를 구별하는 대신, 체중 변동의 **크기**에 집중하고자 합니다. 
- 이를 위해, 체중이 감소한 경우에도 그 **변동 크기**를 양수로 변환하여 체중이 얼마나 많이 변동했는지 확인할 수 있도록 전처리를 진행했습니다.
- 이렇게 하면 체중 변동이 큰 사람들을 분석할 때 도움이 될 수 있습니다.''')


# 1. 주중 수면 시각 처리
train_final["주중 수면 시작 시각"] = train_final["(만12세이상) 주중 잠자리에 든 시각_시"]
train_final.loc[train_final["주중 수면 시작 시각"] == 99, "주중 수면 시작 시각"] = np.nan
train_final["주중 수면 종료 시각"] = train_final["(만12세이상) 주중 일어난 시각_시"]
train_final.loc[train_final["주중 수면 종료 시각"] == 99, "주중 수면 종료 시각"] = np.nan

# 2. 주중 수면 시간 계산 함수
def calculate_sleep_time(start_time, end_time):
    if pd.isna(start_time) or pd.isna(end_time):
        return np.nan
    elif start_time < end_time:
        return end_time - start_time
    else:
        return (24 - start_time) + end_time

train_final["주중 수면 시간"] = train_final.apply(
    lambda row: calculate_sleep_time(row["주중 수면 시작 시각"], row["주중 수면 종료 시각"]), axis=1
)

# 3. 주말 수면 시각 처리
train_final["주말 수면 시작 시각"] = train_final["(만12세이상) 주말 잠자리에 든 시각_시"]
train_final.loc[train_final["주말 수면 시작 시각"] == 99, "주말 수면 시작 시각"] = np.nan
train_final["주말 수면 종료 시각"] = train_final["(만12세이상) 주말 일어난 시각_시"]
train_final.loc[train_final["주말 수면 종료 시각"] == 99, "주말 수면 종료 시각"] = np.nan

# 4. 주말 수면 시간 계산
train_final["주말 수면 시간"] = train_final.apply(
    lambda row: calculate_sleep_time(row["주말 수면 시작 시각"], row["주말 수면 종료 시각"]), axis=1
)

# 5. 주중, 주말 하루 평균 수면 시간 처리
train_final["주중 하루 평균 수면 시간"] = train_final["(만12세이상) 주중(또는 일하는 날) 하루 평균 수면 시간"]
train_final["주말 하루 평균 수면 시간"] = train_final["(만12세이상) 주말(또는 일하지 않은 날, 일하지 않은 전날) 하루 평균 수면 시간"]
train_final.loc[train_final["주중 하루 평균 수면 시간"] == 99, "주중 하루 평균 수면 시간"] = np.nan
train_final.loc[train_final["주말 하루 평균 수면 시간"] == 99, "주말 하루 평균 수면 시간"] = np.nan

# 6. 결측치 처리
train_final["주중 하루 평균 수면 시간"].fillna(train_final["주중 수면 시간"], inplace=True)
train_final["주말 하루 평균 수면 시간"].fillna(train_final["주말 수면 시간"], inplace=True)


# 맥박 규칙성 여부 값의 분포를 출력
st.write(train_final["맥박 규칙성 여부"].value_counts().sort_index(ascending=False))

# 설명을 리스트 형식으로 출력
st.write('''
- **1**: 보통 '정상'을 나타냄
- **2**: 보통 '비정상'을 나타냄
''')
st.write('''맥박 규칙성의 여부에 따라 **우울증**과 관련이 있는지 그래프로 확인해보았습니다.''')

# 세션 상태 초기화
if 'show_image' not in st.session_state:
    st.session_state['show_image'] = False

# 버튼 클릭 시 이미지와 설명을 불러오는 동작
if st.button('그래프 이미지와 설명 불러오기'):
    st.session_state['show_image'] = True

# 버튼을 누르면 이미지와 설명이 표시됨
if st.session_state['show_image']:
    # 그래프 이미지 불러오기
    image = Image.open('./그래프1.PNG')  # 사용자가 업로드한 이미지 파일 경로
    st.image(image, caption='우울증 여부에 따른 맥박 규칙성 여부 비율', use_column_width=True)
    
    # 설명 출력
    st.write('''
    **결론**:
    - 정상 맥박을 가진 사람들 중 약 80.57%가 우울증이 없는 것으로 나타나고, 19.40%가 우울증이 있는 것으로 확인됩니다.
    - 비정상 맥박을 가진 사람들 중 약 85.56%가 우울증이 없고, 14.44%가 우울증이 있는 것으로 확인됩니다.
    - 정상 맥박을 가진 사람들 중 우울증 환자의 비율이 상대적으로 약간 더 높습니다.
    - 그러나 비정상 맥박을 가진 사람들이 우울증과 상관관계가 있을 가능성은 수치상으로 정상 맥박과 큰 차이를 보이지 않습니다.
    
    따라서 맥박 규칙성 여부와 우울증 간에 명확한 상관관계를 발견하기는 어렵지만, 추가적인 분석을 통해 더 깊이 살펴볼 수 있을거같습니다.
    ''')



# 주어진 피처 선택
tmp_feature_col = [
    "성별", "나이", "월평균 가구총소득", "주관적 건강인지", "활동제한 여부",
    "주당 평균 근로시간", "평소 스트레스 인지 정도", "하루 앉아서 보내는 시간", "1인가구 여부", "1년간 체중 증감",
    "현재 우울증 여부(PHQ)"
]

# 열 이름이 정확한지 확인
missing_cols = [col for col in tmp_feature_col if col not in train_final.columns]
if missing_cols:
    st.write(f"다음 열들이 데이터프레임에 존재하지 않습니다: {missing_cols}")
else:
    # tmp_feature_col의 피처만 선택
    train = train_final[tmp_feature_col]


    # 각 열의 데이터 타입 및 결측값 개수를 하나의 표로 병합하여 표시
    data_info = pd.DataFrame({
        "데이터 타입": train.dtypes,
        "결측값 개수": train.isnull().sum()
    })

    
    st.subheader('''선택된 피처의 데이터 정보 및 결측값 정보''')
    st.write(train.shape)
    st.dataframe(data_info)

    # 결측값이 있는 행을 삭제
    train_cleaned = train.dropna()

    # 결측값 처리 후 데이터 타입과 결측값 개수
    cleaned_data_info = pd.DataFrame({
        "데이터 타입": train_cleaned.dtypes,
        "결측값 개수": train_cleaned.isnull().sum()
    })

    # 결측값 처리 후 데이터 타입 및 결측값 정보
    st.subheader("결측값 처리 후 데이터 타입 및 결측값 정보")
    st.dataframe(cleaned_data_info)
    # 결측값 제거 후 데이터 정보 출력
    st.write('''결측값 처리 후 데이터 정보''')
    st.write(train_cleaned.shape)

    # 결측값 처리 후 데이터의 미리보기 (처음 5개 행 표시)
    st.write("결측값 처리 후 데이터 미리보기:")
    st.dataframe(train_cleaned.head())


from sklearn.preprocessing import MinMaxScaler
import platform

# 한글 폰트 설정
def set_korean_font():
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin':  # MacOS
        plt.rc('font', family='AppleGothic')
    else:  # Linux (colab 등)
        plt.rc('font', family='NanumGothic')
    plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

# 한글 폰트 적용
set_korean_font()


# 스케일링할 변수들
min_scale_col = ["나이", "월평균 가구총소득", "주관적 건강인지", "주당 평균 근로시간", 
                 "1년간 체중 증감", "평소 스트레스 인지 정도", "하루 앉아서 보내는 시간"]

# 스케일링 함수 정의
def apply_minmax_scaler(df, columns):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled

# 데이터프레임 스케일링
train_scaled = apply_minmax_scaler(train, min_scale_col)

# Streamlit에 스케일링 전후의 데이터 시각화 및 설명
st.subheader("모델 성능 향상을 위한 스케일링")
st.write('''스케일링을 진행할 컬럼은 "나이", "월평균 가구총소득", "주관적 건강인지", "주당 평균 근로시간", 
         "1년간 체중 증감", "평소 스트레스 인지 정도", "하루 앉아서 보내는 시간" 총 7개컬럼 입니다.''')


# 세션 상태 초기화
if 'show_before' not in st.session_state:
    st.session_state['show_before'] = False
if 'show_after' not in st.session_state:
    st.session_state['show_after'] = False

# 스케일링 전 박스플롯을 버튼으로 불러오기
if st.button('스케일링 전 데이터 박스플롯 보기'):
    st.session_state['show_before'] = True

if st.session_state['show_before']:
    st.write("스케일링 전 데이터 박스플롯")
    fig_before, ax_before = plt.subplots(figsize=(20, 10))
    sns.boxplot(data=train[min_scale_col], palette="Set2", ax=ax_before)
    ax_before.set_title('Boxplot of Features Before Scaling')
    ax_before.set_xlabel('Features')
    ax_before.set_ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(fig_before)

# 스케일링 후 박스플롯을 버튼으로 불러오기
if st.button('스케일링 후 데이터 박스플롯 보기'):
    st.session_state['show_after'] = True

if st.session_state['show_after']:
    st.write("스케일링 후 데이터 박스플롯")
    fig_after, ax_after = plt.subplots(figsize=(20, 10))
    sns.boxplot(data=train_scaled[min_scale_col], palette="Set2", ax=ax_after)
    ax_after.set_title('Boxplot of Features After Scaling')
    ax_after.set_xlabel('Features')
    ax_after.set_ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(fig_after)

    # 박스플롯 비교 설명
    st.subheader("스케일링 전후 박스플롯 비교")
    st.write('''
    **스케일링 전과 후의 차이점**:
    - 스케일링 전에는 변수들 간의 값 범위가 매우 다르기 때문에, 특정 변수들이 더 큰 영향을 미치는 경향이 있었습니다.
    - 스케일링 후에는 모든 변수의 값이 **0과 1 사이**로 표준화되어, 각 변수의 **상대적 중요도가 동일**해집니다.
    - 이를 통해 **모델이 편향되지 않고** 모든 변수에 대해 고르게 학습할 수 있습니다.
    
    **모델 성능 향상**:
    - 스케일링을 적용하면 경사하강법을 사용하는 모델에서 **수렴 속도가 빨라지고**, 거리 기반 알고리즘에서도 **공정한 거리 계산**이 이루어지므로 모델 성능이 향상될 수 있습니다.
    ''')


from sklearn.preprocessing import OneHotEncoder

# 범주형 피처 목록
categorical_features_indices = [
    "성별", "활동제한 여부", "1인가구 여부",
]

code = '''
# 범주형 피처 목록
categorical_features_indices = [
    "성별", "활동제한 여부", "1인가구 여부",
]

# 스케일링된 데이터가 있다고 가정 (예: train_scaled)
# train_scaled는 기존에 스케일링된 데이터프레임입니다.

def apply_one_hot_encoding(df, categorical_cols):
    # OneHotEncoder 객체 생성 (sparse_output=False로 수정)
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first'는 다중공선성 문제를 방지하기 위해 첫 번째 카테고리를 제거
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    
    # 인코딩된 컬럼 이름 생성
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    
    # 기존 데이터프레임에서 범주형 컬럼 제거 후, 원핫인코딩된 컬럼 합치기
    df = df.drop(columns=categorical_cols)
    df_encoded = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    
    return df_encoded

# 원핫인코딩 적용
df_train = apply_one_hot_encoding(train_scaled, categorical_features_indices)
'''

# 코드를 대시보드에 출력
st.subheader('원-핫 인코딩 코드')
st.code(code, language='python')

def apply_one_hot_encoding(df, categorical_cols):
    # OneHotEncoder 객체 생성 (sparse_output=False로 수정)
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first'는 다중공선성 문제를 방지하기 위해 첫 번째 카테고리를 제거
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    
    # 인코딩된 컬럼 이름 생성
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    
    # 기존 데이터프레임에서 범주형 컬럼 제거 후, 원핫인코딩된 컬럼 합치기
    df = df.drop(columns=categorical_cols)
    df_encoded = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    
    return df_encoded

# 원핫인코딩 적용
df_train = apply_one_hot_encoding(train_scaled, categorical_features_indices)

# Streamlit에서 결과 표시
st.subheader("원-핫 인코딩된 데이터프레임")

# 원-핫 인코딩된 데이터프레임을 표로 표시
st.write("원-핫 인코딩된 데이터:")
st.dataframe(df_train)  # 데이터프레임을 Streamlit 대시보드에 표시
# 설명 추가
st.write('''원-핫 인코딩 대상''')
st.write('''범주형 변수인 "성별", "활동제한 여부", "1인가구 여부"에 대하여,
원-핫 인코딩을 진행하여 수치형 변수로 변환 ''')


from sklearn.model_selection import train_test_split

# 주어진 코드를 그대로 스트림릿 대시보드에 출력
code = '''
# y 변수는 "현재 우울증 여부(PHQ)" 컬럼 값만 담습니다.
y = df_train["현재 우울증 여부(PHQ)"]

# X 변수는 "현재 우울증 여부(PHQ)" 컬럼을 제외한 나머지 컬럼들로 구성합니다.
X = df_train.drop(columns=["현재 우울증 여부(PHQ)"])

# train - 데이터셋을 트레인과 테스트로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 우울증 유병 비율 확인
st.subheader('학습 데이터 레이블 값 비율')
y_train_ratio = y_train.value_counts() / y_train.shape[0] * 100
st.write(y_train_ratio)
'''

# 코드를 대시보드에 출력
st.subheader('코드')
st.code(code, language='python')

# 학습 데이터 레이블 값 비율
label_distribution = {
    '우울증 여부': ['0 (우울증 없음)', '1 (우울증 있음)'],
    '비율 (%)': [80.237102, 19.762898]
}

# DataFrame으로 변환
label_df = pd.DataFrame(label_distribution)

# 표 형태로 출력
st.subheader('학습 데이터 레이블 값 비율')
st.table(label_df)  # 표 형태로 출력

# 설명 추가
st.subheader('설명')
st.write('''
1. **데이터셋 분리**:
   - 우리는 **우울증 여부(PHQ)**를 예측하는 **모델을 만들기 위해** 데이터를 **학습 데이터와 테스트 데이터로 분리**합니다.
   - `train_test_split()` 함수를 사용하여 **70%의 데이터는 학습용**으로, **30%의 데이터는 테스트용**으로 나누었습니다.
   - `X`는 **우울증 여부를 제외한 피처(성별, 나이, 소득 등)**를 포함하고, `y`는 **우울증 여부(PHQ)** 정보를 담고 있습니다.
   
2. **레이블 값 비율 확인**:
   - **학습 데이터**의 `y_train`에서 **우울증 여부**가 **1인 사람과 0인 사람의 비율**을 확인한 것입니다.
   - 이 비율을 확인하는 이유는 **데이터가 불균형한지** 여부를 파악하기 위함입니다. **불균형한 데이터**는 머신러닝 모델에서 성능에 영향을 미칠 수 있기 때문입니다.
   
   예를 들어, 만약 **우울증이 있는 사람이 매우 적고**, 우울증이 없는 사람이 대부분이라면, 모델은 우울증이 없는 경우에만 높은 정확도를 보일 수 있습니다. 따라서 **적절한 비율**을 확인하고, 필요할 경우 **데이터 불균형을 해결**하기 위한 방법을 사용할 수 있습니다.

3. **왜 이런 결과가 나왔을까?**:
   - 출력된 비율은 **우울증이 있는 사람(1)**과 **우울증이 없는 사람(0)**의 비율입니다. 만약 **우울증 환자가 상대적으로 적다면**, 이는 일반적으로 **사회적, 환경적 요인**에 따라 우울증의 발생 비율이 낮은 경우일 수 있습니다.
   - 결과적으로 우리는 이 비율을 확인하여 **데이터가 얼마나 불균형한지**를 파악하고, 추후에 이를 개선하기 위한 방법(예: 언더샘플링, 오버샘플링)을 고려할 수 있습니다.
''')

from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# 코드 표시
code = '''
X_test_tmp = X_test.copy()
X_test_tmp["현재 우울증 여부(PHQ)"] = y_test
X_test_tmp["현재 우울증 여부(PHQ)"].value_counts()

X_test_tmp.to_csv("./X_test_tmp.csv")

from imblearn.over_sampling import SMOTE
# SMOTE 적용
smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())
'''

# 코드 표시
st.subheader("SMOTE 적용 코드")
st.code(code, language='python')

# SMOTE 적용 전후 데이터 세트 정보 및 레이블 분포 결과 표시
st.subheader("SMOTE 적용 전후 결과")

# 결과를 표로 보기 좋게 정리
results = {
    '설명': ['SMOTE 적용 전 학습용 피처/레이블 데이터 세트', 'SMOTE 적용 후 학습용 피처/레이블 데이터 세트', 'SMOTE 적용 후 레이블 값 분포(0)', 'SMOTE 적용 후 레이블 값 분포(1)'],
    '값': ['(15099, 10) / (15099,)', '(24230, 10) / (24230,)', 12115, 12115]
}

# 결과를 pandas DataFrame으로 변환
results_df = pd.DataFrame(results)

# DataFrame을 표 형태로 출력
st.table(results_df)

# 설명 추가
st.write('''
- **SMOTE 적용 전**: 학습 데이터의 크기는 **(15099, 10)**, 레이블의 크기는 **(15099,)**입니다.
- **SMOTE 적용 후**: SMOTE 적용 후, 학습 데이터 크기는 **(24230, 10)**, 레이블 크기도 **(24230,)**로 증가했습니다.
- **SMOTE 적용 후 레이블 분포**: 소수 클래스인 1과 다수 클래스인 0의 레이블 개수가 **12115**로 동일해졌습니다. 이는 SMOTE가 소수 클래스를 합성하여 데이터의 불균형을 해결했음을 의미합니다.
''')

st.subheader('로지스틱 회귀')
# 세션 상태 초기화 (각 그래프가 클릭되었는지 여부를 저장)
if 'graph1_shown' not in st.session_state:
    st.session_state['graph1_shown'] = False
if 'graph2_shown' not in st.session_state:
    st.session_state['graph2_shown'] = False
if 'graph3_shown' not in st.session_state:
    st.session_state['graph3_shown'] = False
if 'graph4_shown' not in st.session_state:
    st.session_state['graph4_shown'] = False

# 세션 상태 초기화 (그래프가 표시되었는지 여부를 관리)
if 'graphs_shown' not in st.session_state:
    st.session_state['graphs_shown'] = False

# 버튼 클릭 시 모든 그래프와 설명을 표시
if st.button('모든 그래프 및 설명 보기'):
    st.session_state['graphs_shown'] = True

if st.session_state['graphs_shown']:
    # 첫 번째 그래프 (ROC Curve 및 AUC)
    image1 = Image.open("./그래프2.PNG")
    st.image(image1, caption='ROC Curve 및 AUC')
    st.write('''
    **ROC Curve 및 AUC 설명**:
    - ROC 커브는 모델의 **True Positive Rate(정 양성 비율)**와 **False Positive Rate(위 양성 비율)** 사이의 관계를 시각화한 것입니다.
    - **AUC(Area Under the Curve)** 값은 모델의 성능을 하나의 수치로 나타내는 지표로, **1에 가까울수록 성능이 좋음**을 의미합니다.
    - 여기서 **원본 데이터와 SMOTE 데이터셋**의 AUC 값이 모두 0.801로 동일하게 나타나고 있습니다. 이는 SMOTE를 적용했을 때 모델의 성능이 크게 변하지 않았다는 것을 나타냅니다.
    ''')

    # 두 번째 그래프 (Confusion Matrix 및 Classification Report)
    image2 = Image.open("./그래프3.PNG")
    st.image(image2, caption='Confusion Matrix 및 Classification Report')
    st.write('''
    **혼동 행렬 및 Classification Report 설명**:
    - 혼동 행렬은 **예측된 클래스**와 **실제 클래스**의 분포를 보여줍니다.
    - **정확도, 정밀도, 재현율, F1-score**는 모델 성능을 종합적으로 평가하는 지표들입니다.
    - **SMOTE**를 적용했을 때, 소수 클래스(우울증 있음)에 대한 **재현율(Recall)**이 소폭 상승했지만, 다른 성능 지표는 거의 변동이 없었습니다. 이는 SMOTE를 적용해도 모델이 소수 클래스를 더 잘 식별하지만, 다른 클래스에 대한 성능은 비슷함을 의미합니다.
    ''')

    # 세 번째 그래프 (Feature Importance)
    image3 = Image.open("./그래프4.PNG")
    st.image(image3, caption='Feature Importance')
    st.write('''
    **Feature Importance 설명**:
    - **특성 중요도**는 로지스틱 회귀 모델에서 각 피처가 모델 예측에 얼마나 기여했는지를 나타냅니다.
    - **평소 스트레스 인지 정도**와 **주관적 건강 인지**가 모델에서 가장 중요한 피처로 나타났으며, 이는 두 데이터셋 모두에서 동일합니다.
    - SMOTE를 적용했을 때와 적용하지 않았을 때 특성 중요도 순위는 거의 동일하게 나타났습니다.
    ''')

    # 네 번째 그래프 (Learning Curve)
    image4 = Image.open("./그래프5.PNG")
    st.image(image4, caption='Learning Curve')
    st.write('''
    **Learning Curve 설명**:
    - **러닝 커브**는 모델이 **더 많은 훈련 데이터를 사용함에 따라** 성능이 어떻게 변화하는지를 보여줍니다.
    - 두 그래프 모두에서, **훈련 데이터가 증가**할수록 훈련 정확도는 약간 감소하지만, 교차 검증 점수는 안정적으로 유지됩니다.
    - **SMOTE**를 적용한 데이터셋에서는 약간 더 많은 변동성이 있지만, 전반적인 성능 변화는 크지 않음을 확인할 수 있습니다.
    ''')


st.subheader('XGBoost')
# 세션 상태 초기화 (그래프가 표시되었는지 여부를 관리)
if 'xgboost_graphs_shown' not in st.session_state:
    st.session_state['xgboost_graphs_shown'] = False

# 버튼 클릭 시 XGBoost 관련 모든 그래프와 설명을 표시
if st.button('모든 그래프 및 설명 보기', key='xgboost_button'):
    st.session_state['xgboost_graphs_shown'] = True

if st.session_state['xgboost_graphs_shown']:
    # 첫 번째 그래프 (XGBoost ROC Curve 및 AUC)
    image1 = Image.open("./그래프6.PNG")
    st.image(image1, caption='XGBoost ROC Curve 및 AUC')
    st.write('''
    **XGBoost ROC Curve 및 AUC 설명**:
    - ROC 커브는 XGBoost 모델에서 **True Positive Rate(정 양성 비율)**과 **False Positive Rate(위 양성 비율)** 사이의 관계를 시각화한 것입니다.
    - **AUC 값**은 1에 가까울수록 모델 성능이 좋음을 의미하며, 원본 데이터와 SMOTE 데이터셋에 대한 성능 차이를 확인할 수 있습니다.
    ''')

    # 두 번째 그래프 (XGBoost Confusion Matrix 및 Classification Report)
    image2 = Image.open("./그래프7.PNG")
    st.image(image2, caption='XGBoost Confusion Matrix 및 Classification Report')
    st.write('''
    **XGBoost Confusion Matrix 및 Classification Report 설명**:
    - 혼동 행렬은 예측된 클래스와 실제 클래스의 분포를 보여줍니다.
    - **정밀도, 재현율, F1-score** 등을 통해 XGBoost 모델의 성능을 평가할 수 있으며, SMOTE 적용 전후의 성능 차이를 확인할 수 있습니다.
    ''')

    # 세 번째 그래프 (XGBoost Feature Importance)
    image3 = Image.open("./그래프8.PNG")
    st.image(image3, caption='XGBoost Feature Importance')
    st.write('''
    **XGBoost Feature Importance 설명**:
    - **XGBoost 모델**에서 각 피처가 모델 예측에 얼마나 기여했는지를 나타냅니다.
    - 특정 피처가 예측에 더 큰 영향을 미치고 있음을 확인할 수 있으며, 이는 원본 데이터와 SMOTE 적용 데이터셋 모두에서 비슷하게 나타날 수 있습니다.
    ''')

    # 네 번째 그래프 (XGBoost Learning Curve)
    image4 = Image.open("./그래프9.PNG")
    st.image(image4, caption='XGBoost Learning Curve')
    st.write('''
    **XGBoost Learning Curve 설명**:
    - 러닝 커브는 훈련 데이터 크기가 증가함에 따라 XGBoost 모델의 성능이 어떻게 변하는지를 보여줍니다.
    - **SMOTE 적용 후**에는 소수 클래스에 대한 예측 성능이 향상되었음을 확인할 수 있으며, 특히 데이터가 충분히 많을 때 모델 성능이 더 안정적으로 변합니다.
    ''')


st.subheader('결론 요약')

# 세션 상태 초기화 (결론이 표시되었는지 여부를 관리)
if 'conclusion_shown' not in st.session_state:
    st.session_state['conclusion_shown'] = False

# 버튼 클릭 시 결론 요약 표시
if st.button('서비스 제공', key='conclusion_button'):
    st.session_state['conclusion_shown'] = True

# 결론이 한 번 표시되면 계속 유지되도록 설정
if st.session_state['conclusion_shown']:
    
    st.write('''
    우울증 예측 모델을 활용하여 다양한 **정신 건강 관리 서비스**를 제공할 수 있습니다. 주요 서비스 아이디어는 다음과 같습니다:

    1. **개인 맞춤형 정신 건강 진단 서비스**:
       - 사용자의 데이터를 기반으로 **우울증 위험**을 예측하고, **맞춤형 권장 사항**을 제공할 수 있습니다.

    2. **조기 경고 및 알림 시스템**:
       - 우울증 위험이 높아질 경우 **자동 경고**를 제공하고, **정신 건강 전문가**와의 상담을 연결할 수 있습니다.

    3. **기업용 정신 건강 관리 솔루션**:
       - 직원들의 정신 건강 상태를 관리하고, **복지 프로그램**을 추천하는 기업용 솔루션을 제공합니다.

    이와 같은 서비스는 **정신 건강 관리의 접근성을 높이고**, **우울증의 조기 발견 및 예방**에 기여할 수 있습니다.
    ''')