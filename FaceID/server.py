import base64
from genericpath import isfile
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
from torch.nn import Module
from inference import load_pretrained_model, to_input
from alignment import alignment_procedure
from face_alignment import align
import numpy as np
import time
import cv2
from PIL import Image
import hnswlib
import glob 
import json
from tqdm import tqdm 
import torch

import uvicorn
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import ORJSONResponse, PlainTextResponse
from fastapi.logger import logger
from pydantic import BaseSettings, BaseModel


EMBEDDING_SIZE = 512

idx_mapping_name = {}
model = None
current_no_img = 0

t = hnswlib.Index(space='cosine', dim=EMBEDDING_SIZE)






class Settings(BaseSettings):
    # ... The rest of our FastAPI settings

    BASE_URL = "http://localhost:8001"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"



def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    print(f"Ngrok tunnel opened successful, let's head to {base_url}")




app = FastAPI(title='Deploying a Face Recognition Model with FastAPI')


settings = Settings()
if settings.USE_NGROK:
    # pyngrok should only ever be installed or initialized in a dev environment when this flag is set
    from pyngrok import ngrok

    # Get the dev server port (defaults to 8000 for Uvicorn, can be overridden with `--port`
    # when starting the server
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 8001

    # Open a ngrok tunnel to the dev server
    public_url = ngrok.connect(port).public_url
    logger.info("ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

    # Update any base URLs or webhooks to use the public ngrok URL
    settings.BASE_URL = public_url
    init_webhooks(public_url)



def update_json_logs(new_data, key, filename='logs/logs.json'):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data[key].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)




class JSONObjectPredict(BaseModel):
    file_name: str
    image: str
    keypoints_x1 : int
    keypoints_y1 : int
    keypoints_x2 : int
    keypoints_y2 : int

class JSONObjectRegister(BaseModel):
    user_name: str
    first_img: str
    sec_img: str
    third_img: str

    
@app.get("/")
def home():
    return f"API is working as expected. Now head over to {settings.BASE_URL}/docs."



@app.post("/predict") 
async def prediction(obj: JSONObjectPredict):
    total_start = time.time()
    # Initialization
    global idx_mapping_name, t, model
    response = {}
    obj_dict = obj.dict()

    #################################################
    ################ Reading image ##################
    #################################################
    start = time.time()
    # Write the stream of bytes into a numpy array
    decoded_img = base64.b64decode(obj_dict["image"])
    image_stream = io.BytesIO(decoded_img)
     # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    print("[INFO] Received image shape ({}, {}), left_eye: ({}, {}), right_eye: ({}, {})".format(image.shape[0], image.shape[1], obj_dict["keypoints_x1"], obj_dict["keypoints_y1"], obj_dict["keypoints_x2"], obj_dict["keypoints_y2"]))
    time_stamp = int(time.time())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    end = time.time()
    response["Reading image time"] = " {} ms".format((end-start)*1000)

    #################################################
    ################ Alignment ######################
    #################################################
    start = time.time()

    #aligned_face = alignment_procedure(image, (obj_dict["keypoints_x1"], obj_dict["keypoints_y1"]), (obj_dict["keypoints_x2"], obj_dict["keypoints_y2"]))
    aligned_face = cv2.resize(image, (112, 112))
    end = time.time()
    response["Aligning face time"] = " {} ms".format((end-start)*1000)
    cv2.imwrite(f'logs/images/{time_stamp}.jpg', cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
    #################################################
    ################ Prediction #####################
    #################################################
    start = time.time()
    bgr_tensor_input = to_input(aligned_face)
    feature, _ = model(bgr_tensor_input)
    end = time.time()
    response["Prediction time"] = " {} ms".format((end-start)*1000)


    feature = feature.detach().numpy()
    
    #################################################
    ################ Face search ####################
    #################################################
    start = time.time()
    neighbors = t.knn_query(feature, k = 5)

    most_name_lst = [[idx_mapping_name[str(neighbor)], confidence] for neighbor, confidence in zip(list(neighbors[0][0]), list(neighbors[1][0]))]
    response["Top5"] = [f"{most_name[0]} with {most_name[1]} score" for most_name in most_name_lst]
    most_name_lst = list(filter(lambda x: x[1] < .8, most_name_lst))
    if len(most_name_lst) == 0:
        most_name = "Unknown"
    else:
        most_name_lst = [name[0] for name in most_name_lst]
        most_name = max(set(most_name_lst), key = most_name_lst.count)

    end = time.time()
    response["Face search time"] = " {} ms".format((end-start)*1000)
    response["Result"] = most_name
    response["Timestamp"] = time_stamp
    start = time.time()
    update_json_logs(response, "prediction")
    end = time.time()
    response["Saving logs time"] = " {} ms".format((end-start)*1000)
    total_end = time.time()
    response["Total time"] = " {} ms".format((total_end-total_start)*1000)

    if most_name != "Unknown":
        user_base_img = os.path.join('database/', most_name, most_name+'_0.jpg')
        image =  Image.open(user_base_img)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response["user_image"] = img_str
    else:
        response["user_image"] = ""
    return ORJSONResponse(response)


    

@app.post("/register")
def register(obj: JSONObjectRegister):
    global idx_mapping_name, t, model, current_no_img
    existed = False
    del t
    t = hnswlib.Index(space='cosine', dim=EMBEDDING_SIZE)
    obj_dict = obj.dict()
    name = obj_dict["user_name"]
    directory = os.path.join('database', name)
    file_path = None
    if os.path.isdir(directory):
        no_img = len(os.listdir(directory))
        file_path = os.path.join(directory, name+"_"+str(no_img) + '.jpg')
        existed = True
    else:
        os.mkdir(directory)
        file_path = os.path.join(directory, name+"_0.jpg")

    t.load_index("ANN/db.bin", max_elements = current_no_img+3)
    img_lst = ["first_img", "sec_img", "third_img"]
    for img_str in img_lst:
        # Write the stream of bytes into a numpy array
        decoded_img = base64.b64decode(obj_dict[img_str])
        image_stream = io.BytesIO(decoded_img)
        # Start the stream from the beginning (position zero)
        image_stream.seek(0)
        # Write the stream of bytes into a numpy array
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        cv2.imwrite(file_path, image)
        aligned_rgb_img = align.get_aligned_face(image_path=file_path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        feature = feature.detach().numpy()
        t.add_items(feature, current_no_img)
        idx_mapping_name[str(current_no_img)] = name
        current_no_img += 1

        
    with open("ANN/idx_mapping_name.json", "w") as outfile:
            json.dump(idx_mapping_name, outfile)
            
    index_path='ANN/db.bin'
    print("Saving index to '%s'" % index_path)
    t.save_index(index_path)
    
    return ORJSONResponse({"Status": "Success", "Name": f"{name}", "Existed": existed})

    

def startup(model: Module):
    global t
    representations = {}
    files = glob.glob('database/*/*')
    global current_no_img
    current_no_img = len(files)
    for img_path in tqdm(files):
        print(img_path)
        aligned_rgb_img = align.get_aligned_face(img_path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        embedding, _ = model(bgr_tensor_input)
        name = img_path.split('/')[-1]

        representations[name] = embedding.detach().numpy()

    t.init_index(max_elements = current_no_img, ef_construction = 200, M = 16)
    idx = 0
    idx_mapping_name = {}
    for key, embedding in tqdm(representations.items()):
        t.add_items(embedding, idx)
        idx_mapping_name[idx] = key.split('_')[0]
        idx += 1
    t.set_ef(50)
    index_path='ANN/db.bin'
    print("Saving index to '%s'" % index_path)
    t.save_index(index_path)
    del t
    return idx_mapping_name


def warm_up():
    global model
    for i in range(30):
        feature, norm = model(torch.randn(1,3,112,112))

    









if __name__ == '__main__':
    print("[INFO]Loading model...")
    model = load_pretrained_model("ir_50")
    print("[INFO]Loading done!")
    if not os.path.isfile('ANN/db.bin'):
        idx_mapping_name = startup(model)
        with open("ANN/idx_mapping_name.json", "w") as outfile:
            json.dump(idx_mapping_name, outfile)
    else:
        with open("ANN/idx_mapping_name.json") as json_file:
            idx_mapping_name = json.load(json_file)
        current_no_img = len(idx_mapping_name)
        t.load_index("ANN/db.bin", max_elements = current_no_img)
        warm_up()
        uvicorn.run(app, host="127.0.0.1", port=8001)





 # https://github.com/nmslib/hnswlib