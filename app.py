import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

# Lazy loading
ort_session = None

def get_session():
    global ort_session
    if ort_session is None:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        ort_session = ort.InferenceSession("efficientnet_final.onnx", sess_options)
    return ort_session

classes = ['Audi', 'HyundaiCreata', 'MahindraScorpio', 'RollsRoyce', 'Swift', 'TataSafari', 'ToyotaInnova']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Solo imágenes")

        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))

        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, 0)

        session = get_session()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        ort_out = session.run([output_name], {input_name: x})

        pred_idx = int(np.argmax(ort_out[0][0]))
        conf = float(np.max(ort_out[0][0]))

        return {
            "prediccion": classes[pred_idx],
            "confianza": f"{conf:.2%}",
            "index": pred_idx
        }

    except Exception as e:
        raise HTTPException(500, str(e))





@app.get("/")
def root():
    return {"status": "API ONNX funcionando 🚀"}
