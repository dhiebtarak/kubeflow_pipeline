Some useful commands :
## to start the minikube 
minikube start 
## to stop the minikube 
minikube stop 
## check the pods, replicas, deployment and services
kubectl get all -n kubeflow 
## to lunch a dashbord for the minikube
minikube dashboard 

minikube config set cpus 4

minikube start --cpus=8 --memory=7796 --disk-size=20g
## install istio for knative
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.17.0/istio.yaml
## Access Services of Type LoadBalancer
In Kubernetes, services with type LoadBalancer try to expose an external IP. In cloud setups (like GKE, EKS), a cloud provider assigns that IP.

But in Minikube (a local cluster), there's no cloud provider, so LoadBalancer services won’t work by default.

👉 minikube tunnel simulates this behavior by assigning external IPs locally and routing traffic correctly.

minikube tunnel
## useful paper for istio gateway service 
https://harsh05.medium.com/understanding-ingress-gateway-in-istio-a-detailed-guide-9ee300b9da65

## to create the cluster , this is the commands :

# 1. Create Kind cluster (if not exists)
kind create cluster --name kubeflow --config kind.yaml

# 2. Save kubeconfig
kind get kubeconfig --name kubeflow > /tmp/kubeflow-config
export KUBECONFIG=/tmp/kubeflow-config

# 3. Log in to Docker (if using private images)
docker login

# 4. Create registry secret
kubectl create secret generic regcred \
  --from-file=.dockerconfigjson=$HOME/.docker/config.json \
  --type=kubernetes.io/dockerconfigjson

# 5. Clone Kubeflow manifests (if needed)
git clone https://github.com/kubeflow/manifests.git
cd manifests

# 6. Install Kubeflow
while ! kubectl kustomize example | kubectl apply --server-side --force-conflicts -f -; do
  echo "Retrying...";
  sleep 20;
done

# 7. Access dashboard
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80

## to start the kind cluster 

kubectl config use-context kind-kubeflow

## to extract the pods that are not running 

kubectl get pods --all-namespaces | grep -v Running
## to install kiali 

kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.25/samples/addons/kiali.yaml

knative-eventing   eventing-controller-67b545dfd-xkjxs                      1/1     Running            3 (66s ago)    19h
knative-eventing   eventing-webhook-66844d469d-pjjmg                        1/1     Running            3 (65s ago)    19h
knative-eventing   job-sink-558cb74778-5h92r                                1/1     Running            5 (61s ago)    19h
knative-serving    activator-5f95966686-cpdwn                               2/2     Running            2 (18h ago)    19h
knative-serving    autoscaler-79c8f8b744-tv56p                              2/2     Running            2 (18h ago)    19h
knative-serving    controller-77968c8b7f-7klnt                              2/2     Running            2 (18h ago)    19h
knative-serving    net-istio-controller-fc8b9c9cb-ht7dn                     1/1     Running            2 (18h ago)    19h
knative-serving    net-istio-webhook-58f6fbc8fb-7mqvx                       2/2     Running            2 (18h ago)    19h
knative-serving    webhook-6d6ddd75c-xt2kh                                  2/2     Running            2 (18h ago)    19h
kserve-controller-manager-7cfcfd6d6b-9wlhb               2/2     Running   4 (23h ago)     23h
kserve-localmodel-controller-manager-655ccdf64-n2264     2/2     Running   2 (23h ago)     23h
kserve-models-web-app-6d4cb46c67-9tjrj                   2/2     Running   2 (23h ago)     23h