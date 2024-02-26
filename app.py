from flask import Flask, jsonify, request
from flask_cors import CORS

import numpy as np
import data_utils

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return "Hello!"


# cache
all_layer_neuron_activations = dict()


@app.route("/get_highly_activated_images", methods=["POST", "GET"])
def get_highly_activated_images():
    req = request.get_json()

    layer = req["layer"]
    neurons = req["neurons"]
    model = req["model"]
    print(f"grabbing layer={layer}, neurons={neurons}")

    # 1. Get neuron activations of the right layer
    key = f"{model}-{layer}"
    if key in all_layer_neuron_activations:
        neuron_activations = all_layer_neuron_activations[key]
    else:
        # neuron_activations = np.load(f"./my_data/neuron_activations_{layer}.npy")
        fn = f"my_data/{model}/neuron_activation_image_argsort_{layer}.npy"
        neuron_activations = np.load(fn)
        all_layer_neuron_activations[key] = neuron_activations

    neuron_activations_selected = neuron_activations[neurons]

    # 4. get file names of highly activated images
    image_data = data_utils.get_data("imagenet_val")
    image_fns = [i[0] for i in image_data.imgs]
    activated_image_fns = [
        [image_fns[i] for i in top] for top in neuron_activations_selected
    ]
    ## for imagenet, grab the last 2 in file path and prepend a relative path to it:
    activated_image_fns = [
        ["static/images/" + "/".join(fn.split("/")[-2:]) for fn in act]
        for act in activated_image_fns
    ]

    # 5. return image file names
    return activated_image_fns


if __name__ == "__main__":
    app.run(host='0.0.0.0')
