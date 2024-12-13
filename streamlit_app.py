#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1E5FEhhIRKAKe4hJ_HkdLw6vzCbIAS4Is'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_container_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://i.ibb.co/CVmzvVm/1.jpg",
            "https://i.ibb.co/S71NP5k/3.jpg",
            "https://i.ibb.co/2cxnPy3/image.jpg"
        ],
        'videos': [
            "https://youtu.be/EjrNbUjBdVE?si=UO8P_OIxtLBiKqnH",
            "https://youtu.be/mcbKTiUAkdc?si=JqSvglGhe4qBlbZq",
            "https://youtu.be/6TB7MdZM2uE?si=USPS8fp2Uqoxp8Pb"
        ],
        'texts': [
            "2024 KBO MVP수상 타율 0.347,109타점 40도루를기록하며 팀의 통합우승에 크게 기여함 김도영의 성적은 OPS 1.067,WAR 8.32로 시즌 촤고 수준이였음 리그를 지배한 퍼포먼스를 보여줌",
            "최연소 및 최소 경기 30-30 클럽 달성 김도영은 전반기 20-20 클럽에 가입한뒤 리그 역사상 촤연소이자 최소 경기로 30-30클럽에 이름을 올림김도영은 뛰어난 파워와 주루능력을 동시에 보여준 기록으로 리그의 주목을 받음",
            "역대 최초 월간 10-10클럽및 다양한 신기록 김더영은 역대 최초로 한달 동안 홈런과 도루를 각각 10개이상 기록하며 월간 10-10 클럽에 가입 또 시즌중 최연소 사이클링 히트,21세이하 최다 홈런 달성(38개) 단일 시즌 최다득점(143득점)등 다양한 신기록 세움"
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/bQCw7kG/1.jpg",
            "https://i.ibb.co/5nHHJH7/2.jpg",
            "https://i.ibb.co/nQwHPcD/3.jpg"
        ],
        'videos': [
            "https://youtu.be/K-LL7bmheS4?si=5_4QFo2i7uy2CZBY",
            "https://youtu.be/hkH1tFODUDA?si=snRM88-P8l9xfMw-",
            "https://youtu.be/NdjCIgLAMZE?si=I0tb42Nnv21oU6Up"
        ],
        'texts': [
            "신인시즌의 안정적인 성과로 특히 주요 경기에서 위기 관리 능력을 보여주며 선발투수로 자리를 잡았음 두산 베어스 상대로무실점 호투를 기록하며승리를 이끈게 윤영철의 주요 업적 중 하나",
            "한국 시리즈 대비 준비로 시즌 막바지에 1군으로 복귀한 윤영철은 안정적인 투구로팀의 포스트 시즌 진출에 기여하며 한국시리즈 엔트리 합류를 목표로 준비하고 있음 이는 팀내에서의 윤영철의 가치 입증함",
            "윤영철은 커터를 포함한 다양한 구종을 성공적으로 활용하며 타자를 제압하는 능력을 보여줌 이는 신인으로서의 성장 가능성을 더 높이는 요소가 됨"
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/g4Bht94/1.jpg",
            "https://i.ibb.co/9TMkMf5/2.jpg",
            "https://i.ibb.co/8j2Jm69/3.jpg"
        ],
        'videos': [
            "https://youtu.be/wBkl8l2fNfo?si=MPMMdhDADwdiHbpk",
            "https://youtu.be/tEncTHM34lw?si=vF_QrighCa3iAHtK",
            "https://youtu.be/K8_cbyQPf9U?si=cyTvXi_7B1meoeaE"
        ],
        'texts': [
            "이의리는2021년에 기아 타이거즈의 1차 지명을 받고 입단한 후 그해 4승5패,평균자책점3.61의 준수한 성적으로 리그 신인왕을 수상함 이는 타이거즈 구단에서 36년만에 나온 신인왕 수상자이며 구단 역사상 두번째엿다",
            "이의리는 도쿄 올림픽과WBC에서 대한민국 국가대표팀의 주축 투수로 활약함 특히  올림픽에서 150km/h 를 넘나드는 직구와 강력한 커브로 국제 무대에서도 자신의 존재감을 드러냄",
            "이의리는 2024년 시즌 초 팔꿈치 부상으로 수술을 받으며 시즌 대부분을 쉬었지만 퓨처스리그에서 재활을 거치며 복귀를 준비함 이의리는 복귀는 팀의 선발전 안정화를 위해 매우 중요한 역할로 평가받음"
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

