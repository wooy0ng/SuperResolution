from flask import Flask, request
import json
import pandas as pd
import utils

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

app = Flask(__name__)

# resolution API Route
@app.route("/")
def index():
    return "flask server"


@app.route("/train", methods=['POST'])
def train():
    receive = request.get_json()

    # model 
    data = []
    for json in receive['imgs']:
        data.append(json)
    df = pd.DataFrame(data)
    imgs = [utils.base64_to_img(x) for x in df.data]  # decoded
    
    #### result
    # imgs == super resolution image

    
    df.result = [utils.img_to_base64(img, _type) for img, _type in imgs]
    
    
    _json = df.to_json(orient="table", index=False)
    # response test
    return _json

@app.route("/test")
def test():
    # response test
    return {"response": "test"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=3001)