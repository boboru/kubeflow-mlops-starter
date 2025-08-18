import asyncio
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

request = [
    {
        "userid": ["884208"],
        "cms_segid": ["33"],
        "cms_group_id": ["4"],
        "final_gender_code": ["2"],
        "age_level": ["4"],
        "pvalue_level": ["3"],
        "shopping_level": ["3"],
        "occupation": ["2"],
        "new_user_class_level": ["2"],
        "adgroup_id": ["652419"],
        "cate_id": ["1155"],
        "campaign_id": ["384153"],
        "customer": ["37757"],
        "brand": ["255616"],
        "pid": ["1"],
        "btag": ["1"],
        "price": [0.1474],
    }
]


async def main():
    # Via InferenceRESTClient
    headers = {
        "Host": "dcn-v2.kubeflow-user-example-com.example.com",
        "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImktV3dqSXhNaTdZWVZpN2t3SzdjSUlFZGsxcnlrTGlKMUF1MUsydnp1SW8ifQ.eyJhdWQiOlsiaXN0aW8taW5ncmVzc2dhdGV3YXkuaXN0aW8tc3lzdGVtLnN2Yy5jbHVzdGVyLmxvY2FsIl0sImV4cCI6MTc1NTQ0NDg4MywiaWF0IjoxNzU1MzU4NDgzLCJpc3MiOiJodHRwczovL2t1YmVybmV0ZXMuZGVmYXVsdC5zdmMiLCJqdGkiOiIxYzE2MmEyYi0yNjgxLTQ0OTktOTViMS00ZGQ3OTg2MTc1YzQiLCJrdWJlcm5ldGVzLmlvIjp7Im5hbWVzcGFjZSI6Imt1YmVmbG93LXVzZXItZXhhbXBsZS1jb20iLCJzZXJ2aWNlYWNjb3VudCI6eyJuYW1lIjoiZGVmYXVsdC1lZGl0b3IiLCJ1aWQiOiIzZjdjOTEwZC04ZGNiLTRmZjQtYmM3Ny1mMjFkMjAzMjYwZWUifX0sIm5iZiI6MTc1NTM1ODQ4Mywic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50Omt1YmVmbG93LXVzZXItZXhhbXBsZS1jb206ZGVmYXVsdC1lZGl0b3IifQ.w05jWBPA20LiUHAVzAvWzU9hM2qIGrdhWq1-_kqC3K6tc4rtjhxjp9v5qa2Ly8V_sXaDHjP5uaXUxOuaSwPDHz9ttc54l70UM1-hikO3Gw6p_3ttc1GoCPxpYM4zu78dlXRrTI7a9Y_7h5S0xlKOMSxJ0KV8aIskY2XqweIM1Ee5wS-u_ZyRCijvQKlohvx3pKlSnr-XuetcFmQkPv-fj3azzsYOaNcIPyC-q8tvt40XRE89iMYy-NiP0BroAo2iS-0zI121mbgGCPd3cZSaBhpisXnLBKUsFeBHOVX_D_ARJHRYooZOzVoW4RLA9ElaHHk4ITC71ZnuKxNQt8Fcww",
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
    asyncio.run(main())
