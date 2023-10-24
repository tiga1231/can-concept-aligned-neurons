from flask import Flask, jsonify, request
from flask_cors import CORS

import numpy as np
import data_utils

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return "Hello!"


@app.route("/get_highly_activated_images", methods=["POST", "GET"])
def get_highly_activated_images():
    req = request.get_json()

    layer = req["layer"]
    neurons = req["neurons"]
    print(f"grabbing layer={layer}, neurons={neurons}")

    # 1. Get neuron activations of the right layer
    neuron_activations = np.load(f"./my_data/neuron_activations_{layer}.npy")
    # 2. grab neuron activations in the query
    neuron_activations_selected = neuron_activations[:, neurons]

    # TODO pre-argsort top m
    # # 3. grab highly activated images
    top = 40  # images per neuron
    top_activated_images = np.argpartition(neuron_activations_selected, -top, axis=0)[
        -top:, :
    ].T
    orders = np.argsort(top_activated_images).tolist()
    print(top_activated_images)
    top_activated_images = [
        top[order[::-1]] for top, order in zip(top_activated_images, orders)
    ]
    # print(top_activated_images.shape) # ==[top, n_neurons_selected]

    # 4. get file names of highly activated images
    image_data = data_utils.get_data("imagenet_val")
    image_fns = [i[0] for i in image_data.imgs]
    activated_image_fns = [
        [fn for i, fn in enumerate(image_fns) if i in top]
        for top in top_activated_images
    ]
    ## for imagenet, grab the last 2 in file path and prepend a relative path to it:
    activated_image_fns = [
        ["static/images/" + "/".join(fn.split("/")[-2:]) for fn in act]
        for act in activated_image_fns
    ]

    # 5. return image file names
    return activated_image_fns


if __name__ == "__main__":
    app.run()
