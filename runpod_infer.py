'''
RunPod | Transformer | Handler
'''
import argparse
import io

import torch
import runpod
from runpod.serverless.utils.rp_validator import validate
from transformers import AutoProcessor, BarkModel
import scipy
from flask import Flask, Response


torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True
    },
    # 'do_sample': {
    #     'type': bool,
    #     'required': False,
    #     'default': True,
    #     'description': '''
    #         Enables decoding strategies such as multinomial sampling,
    #         beam-search multinomial sampling, Top-K sampling and Top-p sampling.
    #         All these strategies select the next token from the probability distribution
    #         over the entire vocabulary with various strategy-specific adjustments.
    #     '''
    # },
    # 'max_length': {
    #     'type': int,
    #     'required': False,
    #     'default': 100
    # },
    # 'temperature': {
    #     'type': float,
    #     'required': False,
    #     'default': 0.9
    # }
}


def generator(job):
    '''
    Run the job input to generate text output.
    '''
    # Validate the input
    val_input = validate(job['input'], INPUT_SCHEMA)
    if 'errors' in val_input:
        return {"error": val_input['errors']}
    val_input = val_input['validated_input']

    # input_ids = tokenizer(val_input['prompt'], return_tensors="pt").input_ids.to(device)

    # gen_tokens = model.generate(
    #     input_ids,
    #     do_sample=val_input['do_sample'],
    #     temperature=val_input['temperature'],
    #     max_length=val_input['max_length'],
    # ).to(device)

    # gen_text = tokenizer.batch_decode(gen_tokens)[0]

    
    voice_preset = "v2/en_speaker_6"

    # text_prompt = """
    #     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
    #     But I also have other interests such as playing tic tac toe.
    # """

    text_prompt = """
        Hello
    """

    inputs = processor(text_prompt, voice_preset=voice_preset)

    audio_array = model.generate(**inputs).to(device)
    audio_array = audio_array.cpu().numpy().squeeze()

    sample_rate = model.generation_config.sample_rate

    output = io.BytesIO()
    scipy.io.wavfile.write(output, rate=sample_rate, data=audio_array)
    output.seek(0)

    return 'Done'
    # return Response(output, content_type="audio/wav")
    # scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)



if __name__ == "__main__":
    
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")

    runpod.serverless.start({"handler": generator})