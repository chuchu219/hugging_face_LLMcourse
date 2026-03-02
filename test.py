import gradio as gr

def echo(img):
    return img

iface = gr.Interface(
    fn=echo,
    inputs=gr.Image(
        tool='sketch',        # 允許用滑鼠畫圖
        image_mode='RGB',     # 彩色畫布，改成 'L' 是灰階畫布
        source="canvas",      # 來源設定成空白畫布
        shape=(280, 280),     # 畫布大小
    ),
    outputs="image"
)

iface.launch()
