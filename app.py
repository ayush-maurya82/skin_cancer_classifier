from flask import Flask, jsonify, render_template
from flask_cors import CORS,cross_origin
import PIL
import torch
from torch import nn
from torchvision import transforms, models
from collections import OrderedDict

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*"
    }
})

@app.route('/')
def home():
    return render_template('index.html')

def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(512, 400)),
                      ('relu', nn.ReLU()),
                      ('fc2', nn.Linear(400, 2)),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))

    model.load_state_dict(ckpt, strict=False)

    return model

SAVE_PATH = 'res18_102.pth'

def get_model():
    global model
    model = load_ckpt(SAVE_PATH)
    print("model loaded successfull!")

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

def preprocess_image(image):
    im = PIL.Image.open(image)
    return test_transforms(im)



print("Loading pytorch model....")
get_model()

def predict(image_path, model):
    model.eval()
    img_pros = preprocess_image(image_path)
    img_pros = img_pros.view(1,3,224,224)
    with torch.no_grad():
        output = model(img_pros)
    return output

@app.route("/pred", methods=["POST"])
@cross_origin()
def pred():
    img = request.files['img']
    prediction = ((torch.exp(predict(img, model))).tolist())
    response = {
            'prediction':{
                    'benign':prediction[0][0],
                    'malignant':prediction[0][1]
                    }
            }
    return jsonify(response)

if __name__ == "__main__":
    app.run()
