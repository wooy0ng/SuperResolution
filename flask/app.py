from flask import Flask, request
import json
import pandas as pd
import utils
import torch
import sys
import os

from handler import inference

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"




model_handler = inference.ESRGANHandler('handler/model/RealESRGAN_x4plus.pth')

app = Flask(__name__)

# resolution API Route
@app.route("/")
def index():
    return "flask server"


@app.route("/train", methods=['POST'])
def train():
    print('Training..', file=sys.stderr)
    print(torch.cuda.is_available(), file=sys.stderr)
    receive = request.get_json()

    # model 
    data = []
    for json in receive['imgs']:
        data.append(json)
    df = pd.DataFrame(data)
    imgs = [utils.base64_to_img(x) for x in df.data]  # decoded
    
    #### result
    # imgs == super resolution image
    imgs = [(model_handler.handle(img), _type) for img, _type in imgs]

    
    df.result = [utils.img_to_base64(img, _type) for img, _type in imgs]
    
    _json = df.to_json(orient="table", index=False)
    # response test
    return _json

@app.route("/test")
def test():
    # response test
    return {"response": "test"}

if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug=True, port=3001)
