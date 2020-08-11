# Imbalanced_Image_Classification_With_Complement_Cross_Entropy_Pytorch
**[Yechan Kim](github.com/unique-chan), [Yoonkwan Lee](github.com/brightyoun), and [Moongu Jeon]()**

## Novelty
- **암호화**된 대중교통 이용 데이터에서 어떻게 **관광객의 통행 기록**을 **추출**할 것인가?
- 자세한 알고리즘은 [논문](https://doi.org/10.5626/KTCP.2020.26.8.349)을 참고하시기 바랍니다.
- 저자
  - 김예찬(yechankim@gm.gist.ac.kr), 지스트 전기전자컴퓨터공학부 
  - 김철수(kimcs@jejunu.ac.kr), 제주대학교 전산통계학과
  - 김성백(sbkim@jejunu.ac.kr), 제주대학교 컴퓨터교육과

## Prerequisites
- Python 3.7.4
- Pandas 0.25.1

## Cautions
- 이 알고리즘은 [제주 빅데이터 센터](https://bc.jejudatahub.net/main)에서 제공하는 제주 대중교통 버스 교통카드 빅데이터(tb_bus_user_usage, 버스 이용 데이터)에 최적화되어 있습니다.
- 만약 제주 빅데이터 센터에서 제공하는 데이터 파일의 스키마가 변경될 경우, # 컬럼 상수 하단의 코드를 수정하면 됩니다. 예로, 'user_id'라는 필드가 'bus_user_id'로 변경된 경우, 
~~~
user_id = 'bus_user_id'
~~~

## How to use
- 임의로 디렉토리를 생성한 뒤, 다음과 같이 1개 이상의 제주 대중교통 버스 교통카드 데이터 파일(확장자: csv)을 배치합니다.
- 참고로, 제주 빅데이터 센터는 일일단위로 버스 교통카드 이용 데이터 파일을 제공합니다. 
- 하단과 같이 파일에 이름을 부여하면 직관적일 것입니다. 다만, 파일 이름은 중복만 되지 않는다면, 분석에 어떠한 상관도 없습니다.
- 단, 논문에 제시한 알고리즘 근거에 따라 1년을 초과하는 기간 범위의 데이터 파일들의 배치는 잘못 분석될 가능성이 높습니다.
~~~
tb_bus_user_usage_190601.csv
tb_bus_user_usage_190602.csv
tb_bus_user_usage_190603.csv
... 중략 ...
~~~
- path 변수에 디렉토리의 주소를 삽입합니다.
~~~
# 예로, d 드라이브 밑 tb_bus_user_usage 디렉토리에 분석할 데이터 파일(을 저장한 경우,
### (1) 이하 전처리
path = 'd:/tb_bus_user_usage'
~~~
- 알고리즘의 세부적인 사항을 필요하면 수정합니다.
~~~
# 예로, 연속으로 15일 미만이 아니라, 10일 미만인 버스 이용자를 필터링하고자 할 경우,
... 중략 ...
# 관광객 연속 체류 기간 ('day'의 변수 값(int형)만 수정하면 됩니다! 15 -> 10)
day = 10 
... 중략 ...
### (2) 이하 추출 ①
... 중략 ...
U = list(date_cnt[date_cnt[base_date] < day][user_id].unique())
... 중략 ...
### (3) 이하 추출 ②
... 중략 ...
U2 = list(M2_2[M2_2['diff'] < '%d days' % day][user_id].unique()) 
... 중략 ...
~~~
- 알고리즘을 실행합니다. 알고리즘은 **관광객으로 추정된 버스 이용자의 USER_ID를 추출**하여 **U3 변수**에 저장합니다.
- U3 변수에 담긴 각 USER_ID에 대응하는 버스 이용자의 통행 기록을 분석하면 됩니다.

## Notice
- 본 소스코드를 이용하여 수행한 연구 결과를 논문이나 보고서 등의 형태의 산출물로 게재할 경우, 그 산출물에 하단 레퍼런스를 반드시 인용해야 합니다.
- 인용 포맷은 게재하는 논문이나 보고서의 규정을 준수하시면 됩니다.
>[국문 예시]
김예찬, 김철수, 김성백, "암호화된 대중교통 교통카드 빅데이터에서의 관광객 O-D 통행패턴 추출 알고리즘: 관광 도시, 제주에의 적용," 정보과학회 컴퓨팅의 실제 논문지, Vol. 26, No. 8, pp. 349-361, 2020.

>[영문 예시]
Yechan Kim, Chul-Soo Kim, and Seong-Baeg Kim, "An Algorithm for Extracting Tourists’ O-D Patterns Using Encrypted Smart Card Data of Public Transportation: Application to Tourist City, Jeju," KIISE Transactions on Computing Practices, Vol. 26, No. 8, pp. 349-361, 2020. (in Korean)

## Acknowledgement
- 본 연구는 과학기술정보통신부 및 정보통신기술진흥센터의 SW중심대학 지원사업(No. 2018-0-01863)으로 수행되었습니다.
- 본 연구를 위해 제주 지역 교통카드 빅데이터를 제공한 JTP-제주특별자치도 빅데이터 센터에 감사의 말씀을 전합니다.
