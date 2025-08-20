import asyncio
import argparse
from kserve.protocol.infer_type import InferInput, InferRequest
from kserve import RESTConfig, InferenceRESTClient

import requests


def create_v2_request(request, model_name=None):
    infer_inputs = []
    parameters = {}
    if len(request) > 0 and isinstance(request[0], dict):
        parameters["content_type"] = "pd"
        dataframe = request[0]
        for key, val in dataframe.items():
            infer_input = InferInput(
                name=key,
                shape=[len(val)],
                datatype=(
                    "INT32" if len(val) > 0 and isinstance(val[0], int) else "BYTES"
                ),
                data=val,
            )

            infer_inputs.append(infer_input)

    infer_request = InferRequest(
        model_name=model_name, infer_inputs=infer_inputs, parameters=parameters
    )
    return infer_request

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference against a KServe endpoint")
    parser.add_argument(
        "--token",
        default="",
        help="Bearer token (without the 'Bearer ' prefix)",
    )

    parser.add_argument(
        "--host",
        default="",
        help="Host header value",
    )

    return parser.parse_args()

request = [
    {
        "userid": ["884208", "884207"],
        "cms_segid": ["33", "32"],
        "cms_group_id": ["4", "3"],
        "final_gender_code": ["2", "1"],
        "age_level": ["4", "3"],
        "pvalue_level": ["3", "2"],
        "shopping_level": ["3", "2"],
        "occupation": ["2", "2"],
        "new_user_class_level": ["2", "1"],
        "adgroup_id": ["652419", "652413"],
        "cate_id": ["1155", "1150"],
        "campaign_id": ["384153", "285581"],
        "customer": ["37757", "37756"],
        "brand": ["255616", "255648"],
        "pid": ["1", "1"],
        "btag": ["1", "1"],
        "price": [0.1474, 0.5],
    }
]


async def main(cli_args):
    # Via InferenceRESTClient
    headers = {
        "Host": cli_args.host,
        "Authorization": f"Bearer {cli_args.token}",
    }

    model_name = "dcnv2"
    base_url = "http://localhost:8080/"

    ## v1
    config = RESTConfig(protocol="v1", retries=5, timeout=30)
    client = InferenceRESTClient(config)
    result = await client.infer(
        base_url, {"inputs": request}, model_name, headers=headers
    )
    print("InferenceRESTClient (v1): ", result)

    ## v2
    infer_request = create_v2_request(request=request, model_name=model_name)
    config = RESTConfig(protocol="v2", retries=5, timeout=30)
    client = InferenceRESTClient(config)
    base_url = "http://localhost:8080/"
    result = await client.infer(base_url, infer_request, model_name, headers=headers)
    print("InferenceRESTClient (v2): ", result)

    # Via requests
    ## v1
    base_url = "http://localhost:8080/v1/models/dcnv2:predict"
    payload = {"inputs": request}
    response = requests.post(base_url, headers=headers, json=payload)
    print("Requests (v1): ", response.text)

    ## v2
    base_url = "http://localhost:8080/v2/models/dcnv2/infer"
    response = requests.post(base_url, headers=headers, json=infer_request.to_dict())
    print("Requests (v2): ", response.text)

if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))
