from processor.triton.triton_request_util import triton_get_metadata, set_inout_data
from tools.helper import typetrt2np


class UpscaleModel(object):
    def __init__(self, triton_client, protocol, model_name_up_scale) -> None:

        self.triton_client = triton_client
        self.protocol = protocol
        self.model_name_up_scale = model_name_up_scale

        self.input_name_up_scale, self.output_metadata_up_scale, self.batch_size_up_scale, self.dtype_up_scale = triton_get_metadata(
            protocol, triton_client, model_name_up_scale)

    def transform(self, samples):
        samples = samples.astype(typetrt2np(self.dtype_up_scale))
        inputs, outputs, output_names = set_inout_data(
            self.protocol, self.input_name_up_scale, samples, self.output_metadata_up_scale, self.dtype_up_scale)

        results = self.triton_client.infer(
            self.model_name_up_scale, inputs, outputs=outputs)

        # assert len(
        #     output_names) == 1, "Number of text encoder's outputs must have 1."

        return results.as_numpy(output_names[0])
