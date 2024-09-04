# mychatbot

This project utilizes the OpenVINO™ toolkit to perform inference on deep learning models. The notebook includes code for loading an Intermediate Representation (IR) model, compiling it for a specified device, and running inference using OpenVINO's runtime. The project demonstrates how to integrate OpenVINO with PyTorch and how to tokenize input data using the transformers library. This setup is ideal for deploying optimized models in various applications, including edge and cloud environments.

Prerequisites
Before running the notebook, ensure you have the following libraries installed:

bash
Copy code
pip install openvino-dev transformers torch
Key Commands and Setup
Import OpenVINO Core:

python
Copy code
from openvino.runtime import Core
Load and Compile the IR Model:

python
Copy code
core = Core()
model_ir = core.read_model(model="model_ir/model.xml")
compiled_model = core.compile_model(model=model_ir, device_name="CPU")
Tokenization Using Transformers:

python
Copy code
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
dummy_input = tokenizer("Sample text", return_tensors="pt")
input_ids = dummy_input['input_ids'].numpy()
Inference:

python
Copy code
infer_request = compiled_model.create_infer_request()
infer_request.infer({"input_ids": input_ids})
output = infer_request.get_output_tensor(0).data
print("Inference results:", output)

**Technologies Used : **
OpenVINO™ Toolkit: For optimizing and deploying AI models.
Python: The programming language used for scripting.
PyTorch: Used alongside OpenVINO for model inference.
Transformers Library: Provided by Hugging Face, used for tokenization and handling input data.
