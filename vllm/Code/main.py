from enum import Enum
import json
import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Initialize FastAPI app
app = FastAPI()

num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))


model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct")  # Default to online if env variable missing
llm = LLM(model=model_path, 
          trust_remote_code=True,
          tensor_parallel_size=num_gpus,
          hf_overrides={
              "architectures": ["Qwen2ForCausalLM"]
              },)

class ResponseFormat(BaseModel):
    name : str
    age: int
    

# Generate JSON schema for guided decoding
json_schema = ResponseFormat.model_json_schema()

guided_decoding_params = GuidedDecodingParams(json=json_schema)
sampling_params = SamplingParams(
        top_p=0.9,
        max_tokens=512,
        guided_decoding=guided_decoding_params
        )

# Define request and response models
class PromptRequest(BaseModel):
    prompts: List[str]

class GeneratedResponse(BaseModel):
    results: List[ResponseFormat]

@app.post("/extract", response_model=GeneratedResponse)
def extract_info(request: PromptRequest):
    outputs = llm.generate(prompts=request.prompts, sampling_params=sampling_params)
    # Convert output text to dictionaries
    results = []
    for output in outputs:
        try:
            extracted_data = json.loads(output.outputs[0].text) 
            results.append(ResponseFormat(**extracted_data))  
        except (json.JSONDecodeError, ValueError) as e:
            print(output)
            results.append(ResponseFormat())  
    return {"results": results}
