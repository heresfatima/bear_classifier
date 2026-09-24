import pathlib
import platform
from pathlib import Path
import spaces  # 1. spaces import karein
from fastai.vision.all import load_learner
import gradio as gr

# Windows vs Linux compatibility fix
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'bear_model.pkl'

# Model load karein
learn = load_learner(MODEL_PATH)

# 2. Prediction function ke upar @spaces.GPU lagana zaroori hai
@spaces.GPU
def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Bear Classifier"
)

if __name__ == "__main__":
    demo.launch()