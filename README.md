
### Build the cluster
``` zsh
kind create cluster --name=kubeflow  --config kubeflow-example-config.yaml
```

### Save Kubeconfig
```
kind get kubeconfig --name kubeflow > /tmp/kubeflow-config
export KUBECONFIG=/tmp/kubeflow-config
```

### Create a Secret Based on Existing Credentials to Pull the Images
``` zsh
docker login 

kubectl create secret generic regcred \
    --from-file=.dockerconfigjson=$HOME/.docker/config.json \
    --type=kubernetes.io/dockerconfigjson
```

### Install componenets
- Disable Katib, Pipeline and Spark Operator for saving resources
  
```zsh
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do echo "Retrying to apply resources"; sleep 20; done
```

### Access the Dashboard
``` zsh
export ISTIO_NAMESPACE=istio-system
kubectl port-forward svc/istio-ingressgateway -n ${ISTIO_NAMESPACE} 8080:80
```


For notebook, use other images. Default images are weird...


### Build the Image for Custom Model 
```
pack build --builder=heroku/builder:24 ${DOCKER_USER}/dcnv2:v1
docker push ${DOCKER_USER}/dcnv2:v1
```
             
local testing
```
docker run \
    -e PORT=8080 \
    -p 8081:8080 \
    -v $(pwd):/mnt/models \
    -e MODEL_PATH=/mnt/models/model_weights.pth \
    -e ENCODER_PATH=/mnt/models/preprocess_metadata.pkl \
    -e DENSE_COLS=price \
    -e SPARSE_COLS=userid,cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level,adgroup_id,cate_id,campaign_id,customer,brand,pid,btag \
    ${DOCKER_USER}/dcnv2:v1
```


### Create an `InferenceService`
Use the same pvc and namespace as in the training part. Thus, we can use the training result directly.
Disable istio sidecar injection
```
kubectl apply -f serve.yaml
```


INGRESS_HOST=localhost
INGRESS_PORT=8080
MODEL_NAME=dcnv2
INPUT_PATH=./input.json
SERVICE_HOSTNAME=$(kubectl get inferenceservice -n kubeflow-user-example-com $MODEL_NAME -o jsonpath='{.status.url}' | cut -d "/" -f 3)
TOKEN=$(kubectl create token default-editor -n kubeflow-user-example-com --audience=istio-ingressgateway.istio-system.svc.cluster.local --duration=24h)

```
curl -v -H "Host: $SERVICE_HOSTNAME" -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" -d @$INPUT_PATH http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/$MODEL_NAME:predict 
```

```
hey -z 30s -c 30 -m POST -host ${SERVICE_HOSTNAME}  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" -D $INPUT_PATH http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/$MODEL_NAME:predict
```

[ref](https://github.com/KServe/KServe/tree/master/docs/samples/istio-dex)
