import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from keras.models import load_model

# 1. モデルの読み込み
model_path = "src\keras_model.h5"  # モデルの保存場所に合わせてパスを指定
model = tf.keras.models.load_model(model_path, compile=False)

# 2. 画像の前処理関数
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array.reshape((1, 224, 224, 3))

# 3. 画像判定関数
def predict(image_path, confidence_threshold=0.98):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    confidence_score = prediction[0][class_index]

    class_names = ["golf_wear", "Not_golf_wear"]  # クラス名に合わせて修正
    result_name = class_names[class_index]

    # 4. 判定が自信ない場合は"不確か"を返す
    if confidence_score < confidence_threshold:
        return "不確か", confidence_score
    else:
        return result_name, confidence_score * 100

# 5. Streamlitアプリ
uploaded_file = st.file_uploader("判定したい画像をアップロードしてください。", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    try:
        # 画像を判定
        result_name, confidence_score = predict(uploaded_file)
        
        # 6. 結果を表示
        if result_name == "不確か":
            st.warning(f"判断が自信ないため、{result_name}です (信頼度：{confidence_score * 100:.2f}%)。")
        else:
            st.info(f"この画像は「{result_name}」です (信頼度：{confidence_score * 100:.2f}%)。")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
else:
    st.warning("画像がアップロードされていません。")