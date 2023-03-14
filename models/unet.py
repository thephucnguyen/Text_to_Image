import math
import numpy as np
from einops import repeat

from processor.triton.triton_request_util import triton_get_metadata_multi, set_inout_data_multi


class UnetModel(object):
    def __init__(self, triton_client, protocol,
                 model_name_unet=None,
                 model_name_unet_input_block=None,
                 model_name_unet_middle_block=None,
                 model_name_unet_output_block=None,
                 model_channels=320,
                 is_split_unet=True,
                 dtype=np.float16) -> None:
        if is_split_unet:

            self.inputs_metadata_input_block, self.outputs_metadata_input_block, self.batch_size_input_block = triton_get_metadata_multi(
                protocol, triton_client, model_name_unet_input_block)
            self.inputs_metadata_middle_block, self.outputs_metadata_middle_block, self.batch_size_middle_block = triton_get_metadata_multi(
                protocol, triton_client, model_name_unet_middle_block)
            self.inputs_metadata_output_block, self.outputs_metadata_output_block, self.batch_size_output_block = triton_get_metadata_multi(
                protocol, triton_client, model_name_unet_output_block)

            self.model_name_unet_input_block = model_name_unet_input_block
            self.model_name_unet_middle_block = model_name_unet_middle_block
            self.model_name_unet_output_block = model_name_unet_output_block

            # self.model_name_text_encoder = model_name_text_encoder
        else:
            print(model_name_unet)
            self.inputs_metadata_unet, self.outputs_metadata_unet, self.batch_size_unet = triton_get_metadata_multi(
                protocol, triton_client, model_name_unet)
            self.model_name_unet = model_name_unet

        self.triton_client = triton_client
        self.protocol = protocol
        self.model_channels = model_channels
        self.is_split_unet = is_split_unet
        self.dtype = dtype

    def input_process(self, h, emb, context):
        inputs = (h, emb, context)
        inputs, outputs, output_names = set_inout_data_multi(
            self.protocol, self.inputs_metadata_input_block, inputs, self.outputs_metadata_input_block)
        results = self.triton_client.infer(
            self.model_name_unet_input_block, inputs, outputs=outputs)

        return results.as_numpy(output_names[0]), [results.as_numpy(output_names[i]) for i in range(1, len(output_names))]

    def middle_process(self, h, emb, context):
        inputs = (h, emb, context)
        inputs, outputs, output_names = set_inout_data_multi(
            self.protocol, self.inputs_metadata_middle_block, inputs, self.outputs_metadata_middle_block)

        results = self.triton_client.infer(
            self.model_name_unet_middle_block, inputs, outputs=outputs)

        assert len(output_names) == 1, "Error when "
        return results.as_numpy(output_names[0])

    def output_process(self, hs, h, emb, context):

        inputs = (h, emb, context, *hs)
        # print([type(i) for i in inputs])
        inputs, outputs, output_names = set_inout_data_multi(
            self.protocol, self.inputs_metadata_output_block, inputs, self.outputs_metadata_output_block)

        results = self.triton_client.infer(
            self.model_name_unet_output_block, inputs, outputs=outputs)

        assert len(output_names) == 1, "Error when "
        return results.as_numpy(output_names[0])

    def unet_process(self, h, t, context):
        inputs = (h.astype(self.dtype), t.astype(
            self.dtype), context.astype(self.dtype))

        inputs, outputs, output_names = set_inout_data_multi(
            self.protocol, self.inputs_metadata_unet, inputs, self.outputs_metadata_unet)

        results = self.triton_client.infer(
            self.model_name_unet, inputs, outputs=outputs)

        assert len(output_names) == 1, "Error when "
        return results.as_numpy(output_names[0])

    def apply(self, x, timesteps, context):
        h = x
        if self.is_split_unet:
            t_emb = self.timestep_embedding(
                timesteps, self.model_channels, repeat_only=False)

        # print('t_emb: ', t_emb)
            # start = time.time()
            emb, hs = self.input_process(h, t_emb, context)
            # middle = time.time()
            # print('emb: ', emb)
            # print('h: ', hs[-1])
            h = self.middle_process(hs[-1], emb, context)
            # end = time.time()
            # print('h after middle: ', h)
            out = self.output_process(hs, h, emb, context)
            # print("Time of input block: ", middle - start)
            # print("Time of middle block: ", end - middle)
            # print("Time of output block: ", time.time() - end)
        else:
            # start = time.time()
            out = self.unet_process(h, timesteps, context)
            # print('Time of unet: ', time.time() - start)

        return out

    def apply_model(self, x_noisy, t, c_crossattn: list = None, return_ids=False):

        # if isinstance(cond, dict):
        #     # hybrid case, cond is exptected to be a dict
        #     pass
        # else:
        if not isinstance(c_crossattn, list):
            c_crossattn = [c_crossattn]
        #     key = 'c_crossattn'
        #     cond = {key: cond}

        cc = np.concatenate(c_crossattn, 1)
        x_recon = self.apply(x_noisy, t, context=cc)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def timestep_embedding(self, timesteps, dim, max_period=10000, repeat_only=False):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        if not repeat_only:
            half = dim // 2
            freqs = np.exp(
                -math.log(max_period) * np.arange(start=0,
                                                  stop=half, dtype=self.dtype) / half
            )
            args = timesteps[:, None].astype(self.dtype) * freqs[None]
            embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
            if dim % 2:
                embedding = np.concatenate(
                    [embedding, np.zeros_like(embedding[:, :1])], axis=-1)
        else:
            embedding = repeat(timesteps, 'b -> b d', d=dim)
        return embedding.astype(self.dtype)
