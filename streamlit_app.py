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
            "https://i.ibb.co/2cxnPy3/image.jpg",
            "https://i.ibb.co/S71NP5k/3.jpg",
            "https://i.ibb.co/2cxnPy3/image.jpg"
        ],
        'videos': [
            "https://youtu.be/EjrNbUjBdVE?si=UO8P_OIxtLBiKqnH",
            "https://youtu.be/mcbKTiUAkdc?si=JqSvglGhe4qBlbZq",
            "https://youtu.be/6TB7MdZM2uE?si=USPS8fp2Uqoxp8Pb"
        ],
        'texts': [
            "경기 잘함.",
            "잘생김",
            "어리다"
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
            "귀여움.",
            "좌완 투수.",
            "강속구."
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
            "쿼카닮음.",
            "탈삼진.",
            "키큼."
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

