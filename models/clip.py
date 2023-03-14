from processor.triton.triton_request_util import triton_get_metadata, set_inout_data
from transformers import CLIPTokenizer
from tools.helper import typetrt2np


class ClipModel(object):
    def __init__(self,  triton_client, protocol, model_name_tokenizer, model_name_text_encoder) -> None:

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name_tokenizer)

        self.model_name_text_encoder = model_name_text_encoder

        self.input_name_text_encoder, self.output_metadata_text_encoder, self.batch_size_text_encoder, self.dtype_text_encoder = triton_get_metadata(
            protocol, triton_client, model_name_text_encoder)
        self.triton_client = triton_client
        self.protocol = protocol

    def encode(self, text: str, max_length: int):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        np_dtype = typetrt2np(self.dtype_text_encoder)

        tokens = batch_encoding["input_ids"].cpu(
        ).detach().numpy().astype(np_dtype)

        inputs, outputs, output_names = set_inout_data(
            self.protocol, self.input_name_text_encoder, tokens, self.output_metadata_text_encoder, self.dtype_text_encoder)

        results = self.triton_client.infer(
            self.model_name_text_encoder, inputs, outputs=outputs)

        assert len(
            output_names) == 1, "Number of text encoder's outputs must have 1."

        return results.as_numpy(output_names[0])
