import os
from builtins import range
import sys

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import tritonclient.grpc.model_config_pb2 as mc


def parse_model_grpc(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    output_metadata = model_metadata.outputs
    input_config = model_config.input[0]

    input_batch_dim = True

    return (input_metadata.name, output_metadata, model_config.max_batch_size, input_metadata.datatype)


def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    output_metadata = model_metadata['outputs']
    input_config = model_config['input'][0]
    print(input_metadata)
    print(input_config)

    return (input_metadata['name'], output_metadata,
            model_config['max_batch_size'], input_metadata['datatype'])


def parse_model_http_multi(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """

    inputs_metadata_list = model_metadata['inputs']
    output_metadata = model_metadata['outputs']
    input_config = model_config['input']

    input_list = []

    # for i in range(len(input_metadata)):
    #     input_list.append([input_metadata[i]['name'], input_metadata[i]
    #                       ['datatype'], input_metadata[i]['shape'], input_metadata[i]['datatype']])

    return inputs_metadata_list, output_metadata, model_config['max_batch_size']


def triton_get_metadata(protocol, triton_client, model_name):
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    if protocol == "grpc":
        input_name, output_metadata, batch_size, dtype = parse_model_grpc(
            model_metadata, model_config.config)
    else:
        input_name, output_metadata, batch_size, dtype = parse_model_http(
            model_metadata, model_config)

    return input_name, output_metadata, batch_size, dtype,


def triton_get_metadata_multi(protocol, triton_client, model_name):
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    if protocol == "grpc":
        inputs_metadata_list, output_metadata, batch_size, dtype = parse_model_grpc(
            model_metadata, model_config.config)
    else:
        inputs_metadata_list, output_metadata, batch_size = parse_model_http_multi(
            model_metadata, model_config)

    return inputs_metadata_list, output_metadata, batch_size


def init_triton_client(protocol, url, verbose):
    try:
        if protocol == "grpc":
            # Create gRPC client for communicating with the server
            triton_client = grpcclient.InferenceServerClient(
                url=url, verbose=verbose)
        else:
            # Create HTTP client for communicating with the server
            triton_client = httpclient.InferenceServerClient(
                url=url, verbose=verbose)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)
    return triton_client


def get_filenames(image_filename, batch_size):
    filenames = []
    if os.path.isdir(image_filename):
        filenames = [
            os.path.join(image_filename, f)
            for f in os.listdir(image_filename)
            if os.path.isfile(os.path.join(image_filename, f))
        ]
    else:
        filenames = [
            image_filename,
        ]

    filenames.sort()

    if len(filenames) <= batch_size:
        batch_size = len(filenames)
    else:
        print("The number of images exceeds maximum batch size,"
              "only the first {} images, sorted by name alphabetically,"
              " will be processed".format(batch_size))

    return filenames, batch_size


def set_inout_data(protocol, input_name, batched_image_data, output_metadata, dtype):

    # Set the input data
    inputs = []
    if protocol == "grpc":
        inputs.append(
            grpcclient.InferInput(input_name, batched_image_data.shape,
                                  dtype))
        inputs[0].set_data_from_numpy(batched_image_data)
    else:
        inputs.append(
            httpclient.InferInput(input_name, batched_image_data.shape,
                                  dtype))
        inputs[0].set_data_from_numpy(batched_image_data)

    output_names = [
        output.name if protocol == "grpc" else output['name']
        for output in output_metadata
    ]

    outputs = []
    for output_name in output_names:
        if protocol == "grpc":
            outputs.append(
                grpcclient.InferRequestedOutput(output_name))
        else:
            outputs.append(
                httpclient.InferRequestedOutput(output_name))

    return inputs, outputs, output_names


def set_inout_data_multi(protocol, inputs_metadata_list, batched_input_list, output_metadata):

    # Set the input data
    inputs = []
    # print('Input metadata: ', inputs_metadata_list)
    assert len(inputs_metadata_list) == len(
        batched_input_list), f"Warning: Got {len(inputs_metadata_list)} input metadata but number of batched is {len(batched_input_list)}"
    for idx in range(len(inputs_metadata_list)):
        if protocol == "grpc":
            inputs.append(
                grpcclient.InferInput(inputs_metadata_list[idx]['name'], batched_input_list[idx].shape,
                                      inputs_metadata_list[idx]['datatype']))
        else:
            # print(
            #     'Multi: ', inputs_metadata_list[idx]['name'], batched_input_list[idx].shape)
            inputs.append(
                httpclient.InferInput(inputs_metadata_list[idx]['name'], batched_input_list[idx].shape,
                                      inputs_metadata_list[idx]['datatype']))
            # print(batched_input_list[idx].dtype,
            #       inputs_metadata_list[idx]['datatype'])
            inputs[idx].set_data_from_numpy(batched_input_list[idx])

    output_names = [
        output.name if protocol == "grpc" else output['name']
        for output in output_metadata
    ]

    outputs = []
    for output_name in output_names:
        if protocol == "grpc":
            outputs.append(
                grpcclient.InferRequestedOutput(output_name))
        else:
            outputs.append(
                httpclient.InferRequestedOutput(output_name))

    return inputs, outputs, output_names


def set_in_tgt(protocol, input_list, batched_image_data):

    # Set the input data
    # print(input_list)
    inputs = []
    if protocol == "grpc":
        inputs.append(
            grpcclient.InferInput(input_list[0][0], batched_image_data.shape,
                                  input_list[0][1]))
        inputs[0].set_data_from_numpy(batched_image_data)
    else:
        inputs.append(
            httpclient.InferInput(input_list[0][0], batched_image_data.shape,
                                  input_list[0][1]))
        inputs[0].set_data_from_numpy(batched_image_data)

    return inputs
