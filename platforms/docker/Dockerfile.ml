# The image that carries the models. One image, four platforms.
#
# Examples 05 (RAG) and 06 (distillation) need an embedding model and a language model.
# On Databricks those are hosted serving endpoints. Everywhere else they are ordinary
# open-source weights — and they have to actually BE somewhere.
#
# This image is the answer for Docker, Kubernetes, AWS EMR Serverless and GCP Dataproc
# Serverless alike: all four accept a custom container image, so the same artifact runs
# on all of them. The only thing that differs is which registry it is pushed to.
#
#   docker      docker build -f platforms/docker/Dockerfile.ml -t ubunye-ml .
#   kubernetes  kind load docker-image ubunye-ml   (or push to any registry)
#   AWS EMR     push to ECR,               --release-label + imageUri
#   GCP Dataproc push to Artifact Registry, --container-image
#
# It is built FROM the base image, so there is exactly one definition of the engine,
# Spark, Delta and the JDBC driver, and this adds only the model stack on top.
ARG BASE=ubunye-portable:ci
FROM ${BASE}

USER root

# The CPU wheels, explicitly. The default torch wheel drags in the entire CUDA stack —
# about 2GB of GPU libraries that will never be opened on any of these platforms, and
# that is 2GB every executor pulls before it can start.
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.2" \
      "transformers>=4.44" \
      "sentence-transformers>=3" \
      "accelerate>=0.30"

# Bake the WEIGHTS in, do not fetch them at run time.
#
# A container that downloads a gigabyte from huggingface.co on every start is a
# container that (a) is slow, and (b) silently depends on the internet — which is
# exactly what a locked-down Kubernetes cluster, a VPC-isolated EMR job, or an
# air-gapped runner does not have. The failure would arrive at run time, in the cloud,
# after you had already paid to start the job.
#
# The download happens once, here, where a failure is a build failure.
ENV HF_HOME=/opt/hf
ENV TRANSFORMERS_OFFLINE=0
ARG LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG LOCAL_CHAT_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ARG STUDENT_MODEL=distilbert-base-uncased

RUN python - <<PY
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

SentenceTransformer("${LOCAL_EMBED_MODEL}", device="cpu")

AutoTokenizer.from_pretrained("${LOCAL_CHAT_MODEL}")
AutoModelForCausalLM.from_pretrained("${LOCAL_CHAT_MODEL}")

# The student that example 06 fine-tunes.
AutoTokenizer.from_pretrained("${STUDENT_MODEL}")
AutoModelForSequenceClassification.from_pretrained("${STUDENT_MODEL}", num_labels=3)
print("weights baked in")
PY

# Now that the weights are present, refuse to reach for the network at run time. If a
# model is missing, the job fails immediately and says so, rather than quietly
# downloading it and making the image's contents a lie.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    MODEL_BACKEND=local \
    TORCH_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

ENV LOCAL_EMBED_MODEL=${LOCAL_EMBED_MODEL} \
    LOCAL_CHAT_MODEL=${LOCAL_CHAT_MODEL}
