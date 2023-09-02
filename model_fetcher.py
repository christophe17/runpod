'''
RunPod | Transformer | Model Fetcher
'''

import argparse

import torch
from transformers import AutoProcessor, BarkModel


# def download_model(model_name):

#     processor = AutoProcessor.from_pretrained("suno/bark")
#     model = BarkModel.from_pretrained("suno/bark")


# # ---------------------------------------------------------------------------- #
# #                                Parse Arguments                               #
# # ---------------------------------------------------------------------------- #
# parser = argparse.ArgumentParser(description=__doc__)
# parser.add_argument("--model_name", type=str,
#                     default="gpt-neo-1.3B", help="URL of the model to download.")


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")

    # args = parser.parse_args()
    # download_model(args.model_name)