from processor.triton.triton_request_util import triton_get_metadata, set_inout_data
from tools.helper import typetrt2np


class AutoEncoder(object):
    def __init__(self, triton_client, protocol, model_name_vae_decoder) -> None:
        self.triton_client = triton_client
        self.protocol = protocol
        self.model_name_vae_decoder = model_name_vae_decoder

        self.input_name_vae_decoder, self.output_metadata_vae_decoder, self.batch_size_vae_decoder, self.dtype_vae_decoder = triton_get_metadata(
            protocol, triton_client, model_name_vae_decoder)

    def decode(self, samples):
        dtype = typetrt2np(self.dtype_vae_decoder)
        samples = samples.astype(dtype)

        inputs, outputs, output_names = set_inout_data(
            self.protocol, self.input_name_vae_decoder, samples, self.output_metadata_vae_decoder, self.dtype_vae_decoder)

        results = self.triton_client.infer(
            self.model_name_vae_decoder, inputs, outputs=outputs)

        assert len(
            output_names) == 1, "Number of text encoder's outputs must have 1."

        return results.as_numpy(output_names[0])
