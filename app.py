import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import ImageOps,Image

model = tf.keras.models.load_model('model.h5')

def recognize_digit(image):
    if image is not None:
        # 如果是 numpy array，轉成 PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image = np.array(image).astype('float32') / 255.0
        image = image.reshape((1, 28, 28, 1))
        prediction = model.predict(image)
        
        return {str(i): float(prediction[0][i]) for i in range(10)}
        
    return ''

iface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Image(
        image_mode='L',    # 灰階
        tool='sketch',     # 手繪工具
        source='canvas',   # 空白畫布
        shape=(1000, 500)   # 畫布大小
    ),
    outputs=gr.Label(num_top_classes=3),
    live=True
)

iface.launch()
