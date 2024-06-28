# Rocket.Chat AI Preview Setup Guide

This guide will help you to setup Rocket.Chat AI on your local machine. The project is in beta and we are working on improving the setup process. If you face any issues or have any feedbacks, please reach out to us on the [Rocket.Chat AI channel](https://open.rocket.chat/channel/Rocket-Chat-SAFE-ai-v-hub).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- Instance with a GPU
- NVIDIA GPU(s) with CUDA support
- CPU: x86_64 architecture
- OS: any Linux distros which:
  - - Are [supported by the NVIDIA Container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html)
  - - Have glibc >= 2.35 (see output of ld -v)
- CUDA drivers: Follow the installation guide [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

  > We only support the `cuda` and versions as `12.5`, `12.2` and `12.1`. If you have a different version, please upgrade or downgrade to the supported versions. Otherwise, you can reach out to the Rocket.Chat team for feedback on supporting the version you have.

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)

- Docker with GPU support

  > To test if the GPU is accessible in the docker container, follow the steps listed in the [Compose GPU Support](https://docs.docker.com/compose/gpu-support/)

- [Rocket.Chat License](https://www.rocket.chat/pricing) (Starter, Pro, or Enterprise), Starter license is free for small teams. For more information, please refer to the [Rocket.Chat Pricing](https://www.rocket.chat/pricing) page.

## Recommended Hardware

- Tested on an AWS EC2 instance with the following configuration:
  - Instance type: `g5.2xlarge`
  - vCPUs: 8
  - Memory: 32 GB
  - GPU: NVIDIA A10G
  - VRAM: 24 GB
  - Storage: 450 GB

Minimum requirements:

- vCPUs: 4
- Memory: 12 GB

- GPU VRAM:
- - For `Llama3-8B` model: 8 GB
- - For `Llama3-70B` model: 40 GB

- Storage:
- - For `Llama3-8B` model: 100 GB
- - For `Llama3-70B` model: 500 GB

## Installation

1. Clone/Init the repository
   If using `https`:

```bash
git clone https://github.com/RocketChat/Rocket.Chat.AI.Preview.git
```

For zip download:

```bash
unzip Rocket.Chat.AI.Preview-main.zip -d Rocket.Chat.AI.Preview
```

2. Change the directory

```bash
cd Rocket.Chat.AI.Preview
```

### RAG Pipeline (Rubra.AI)

3. Start Rubra.AI

```bash
docker-compose -f docker-compose.yaml --profile rubra up -d
```

If you're using a newer version of Docker (Docker CLI versions 1.27.0 and newer), you may need to use the following command:

```bash
docker compose -f docker-compose.yaml --profile rubra up -d
```

Once everything is running. To verify that every service is running, you can run the following command:

```bash
docker ps --format "{{.Names}}"
```

Should return the following services:

```bash
$ docker ps --format "{{.Names}}"
ui
api-server
task-executor
vector-db-api
milvus
milvus-minio
mongodb
litellm
milvus-etcd
text-embedding-api
redis
```

In case you don't see any of the services listed above, you can reach out to the Rocket.Chat team for support.

If everything is running, you can now access the Rubra UI at [http://localhost:8501](http://localhost:8501). Now move on to the next step to start the LLM service.

> Note `localhost` is the default hostname. If you are using a different hostname, replace `localhost` with your hostname.

### Deploy LLM

> Note we support two methods to run LLM, one with Docker and the other with helm. For the Docker method, follow the steps below. For scaling and production use cases we recommend using our optimized Docker and Helm Deployments, for access, please reach out to us on the [Rocket.Chat AI channel](https://open.rocket.chat/channel/Rocket-Chat-SAFE-ai-v-hub).
> This deployment only supports 4 concurrent requests to the LLMs.

To verify the installed CUDA version on your system, execute the following command:

```bash
nvidia-smi
```

This command will produce an output similar to the following:

```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.02              Driver Version: 555.42.02      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10G                    Off |   00000000:00:1E.0 Off |                    0 |
|  0%   31C    P0             58W /  300W |       1MiB /  23028MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Additionally, to confirm the version of the CUDA Compiler Driver, use the command:

```
nvcc --version
```

The expected output is as follows:

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Apr_17_19:19:55_PDT_2024
Cuda compilation tools, release 12.5, V12.5.40
Build cuda_12.5.r12.5/compiler.34177558_0
```

In the provided outputs, the `CUDA Version` is identified as `12.5`, and the version of the `CUDA compilation tools` is also `12.5`.

> Should there be a discrepancy in versions, it is recommended to align your system's CUDA version with the supported versions by either upgrading or downgrading.

This version information (excluding the period) is utilized to configure the `PLATFORM_TAG` within the `.env` file.

For instance, with a `CUDA version` of `12.5`, the `PLATFORM_TAG` should be set to `cuda125`.

For example, if it is `12.5`, the `PLATFORM_TAG` should be `cuda125`.

> Note: Supported CUDA versions include `12.5`, `12.2`, and `12.1`. If your system's version does not match any of the supported versions, please consider updating your CUDA installation. Alternatively, for assistance with unsupported versions, contact the Rocket.Chat team for guidance on compatibility.

Start with defining the environment variables in the `.env` file. You can copy the `.env.llm.example` file and rename it to `.env`. Then, modify the following variables as needed:

```bash
# For the model weights
MODEL_NAME=Llama-3-8B-Instruct-q4f16_1-MLC

# For the MLC library
PLATFORM_TAG=cuda125
RELEASE=0.0.1
```

```bash
docker-compose -f docker-compose.yaml --profile mlc-llm up -d
```

If you're using a newer version of Docker (Docker CLI versions 1.27.0 and newer), you may need to use the following command:

```bash
docker compose -f docker-compose.yaml --profile mlc-llm up -d
```

Once the Docker container is running, you can call the LLM API using the following command:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Llama-3-8B-Instruct-q4f16_1-MLC",
        "messages": [
            {"role": "user", "content": "Hello! Our project is MLC LLM. What is the name of our project?"}
        ]
  }' \
  http://localhost:1234/v1/chat/completions
```

> Note `localhost` and `1234` are the default hostname and port. If you are using a different hostname and port, replace `localhost` and `1234` with your hostname and port.

If you get a response, the LLM service is running successfully.

Now you have successfully set up Rocket.Chat AI on your local machine. You can now integrate it with your Rocket.Chat instance.

## Integration

> Note: Replace the service names with the actual hostname and port if you are using a different hostname and port.

1. Go to your Rocket.Chat instance.
2. Install the `Rocket.Chat AI` app from the marketplace. You can find the app by searching for `Rocket.Chat AI` under the (`Admininistration` > `Marketplace` > `Explore`). It's an premium app, so you need to have a valid license (`Starter` or `Pro` or `Enterprise`) to install the app.
3. After installing the app, go to the `Rocket.Chat AI` app settings page (`Admininistration` > `Marketplace` > `Installed` > `Rocket.Chat AI` > `Settings`).
4. Enter the `Model URL` with the LLM API URL. For example, `http://llama3-8b:1234/v1`. (don't include the `/chat/completions` part).
5. Enter the URL of the service `milvus` with the port `19530`. For example, `http://milvus:19530` in the setting "Vector database URL".
6. Enter the URL of the service `text-embedding-api` with the port `8020`. For example, `http://text-embedding-api:8020/embed_multiple` in the setting "Text embedding API URL".
7. For setting up the Knowledge base refer to the [Knowledge base setup video](./assets/rubra_assistant.mp4).
8. For the setting `Vector database collection`, you have two options:

   a. Call the endpoint `http://api-server:8000/assistants` and search for the assistant you want to integrate with.
   Example response looks like:

```json
{
  "object": "list",
  "data": [
    {
      "_id": {},
      "id": "asst_226796",
      "object": "assistant",
      "created_at": 1718975287,
      "name": "Demo Assistant",
      "description": "An assistant for RAG",
      "model": "custom",
      "instructions": "You are a helpful assistant",
      "tools": [
        {
          "type": "retrieval"
        }
      ],
      "file_ids": ["file_0cff17", "file_9b02be"],
      "metadata": {}
    }
  ],
  "first_id": "asst_226796",
  "last_id": "asst_226796",
  "has_more": false
}
```

Now copy the id of the assistant you want to integrate with the Rocket.Chat AI, from the field `id` in the example we have it as `asst_226796`.
Once copied enter the same in the Rocket.Chat AI app settings page in the field `Vector database collection`.

b. You can directly enter the `http://api-server:8000?name=Demo Assistant` in the field `Vector database collection` in the Rocket.Chat AI app settings page. If the assistant existst, it will automatically fetch the assistant and replace the settings with `asst_XYZ` where `XYZ` is the id of the assistant. If the field didn't change, it means the assistant doesn't exist or there is an issue with the API.

> Note: `http://api-server:8000` is the default hostname and port. If you are using a different hostname and port, replace `http://api-server:8000` with your hostname and port.

Once you have integrated the Rocket.Chat AI with your Rocket.Chat instance, you can start using the AI features in your Rocket.Chat instance.

### Bring your own Milvus vector database

If you have your own Milvus vector database, you can use it with the Rocket.Chat AI. You can follow the steps below to integrate your Milvus vector database with the Rocket.Chat AI.

1. Go to your Rocket.Chat instance.
2. Install the `Rocket.Chat AI` app from the marketplace. You can find the app by searching for `Rocket.Chat AI` under the (`Admininistration` > `Marketplace` > `Explore`). It's an premium app, so you need to have a valid license (`Starter` or `Pro` or `Enterprise`) to install the app.
3. After installing the app, go to the `Rocket.Chat AI` app settings page (`Admininistration` > `Marketplace` > `Installed` > `Rocket.Chat AI` > `Settings`).
4. Enter the `Model URL` with the LLM API URL. For example, `http://llama3-8b:1234/v1`. (don't include the `/chat/completions` part).
5. Enter the URL of your Milvus vector database with the port `19530`. For example, `http://milvus:19530` in the setting "Vector database URL".
6. Enter your API Key in the setting `Vector database API key`.
7. Enter the text field where the text data is stored in the collection schema in the setting `Vector database text field`.
8. Enter your embedding model (used when ingesting the data) in the field `Embedding model URL`.

Make sure your Embedding Model URL follows a certain format for request payload and response.

> Input:
>
> ```json
> {
>     [
>         "text1", "text2", ...
>     ]
> }
> ```
>
> Output:
>
> ```json
> {
>     "embeddings": [
>             [0.1, 0.2, 0.3, ...],
>             [0.4, 0.5, 0.6, ...]
>
>     ]
> }
> ```

### About the Config Files

> Note: Once modified, you need to restart the services for the changes to take effect. You can restart the services using the following command:

```bash
docker-compose -f docker-compose.yaml --profile rubra restart
```

1. `llm-config.yaml`: This file contains the configuration for the LLM service of Rubra AI. You can modify the configuration as per your requirements.

```yaml
OPENAI_API_KEY: sk-....X0FUz2bhgyRW32qF1 # OpenAI API key - Enables the use of OpenAI models in Rubra AI
  REDIS_HOST: redis # Redis host
  REDIS_PASSWORD: "" # Redis password
  REDIS_PORT: "6379" # Redis port

model_list:
  - litellm_params:
      api_base: http://host.docker.internal:1234/v1 # LLM API base URL
      api_key: None # LLM API key
      custom_llm_provider: openai # Don't change this for custom models
      model: openai/custom # Model name - must be in the format openai/custom
    model_name: custom
```

2. `milvus.yaml`: This file contains the configuration for the Milvus service of Rubra AI. For more information, refer to the [Milvus documentation](https://milvus.io/docs/configure-docker.md#Modify-the-configuration-file).

### Troubleshooting

1. If following error:

```shell
 ✘ text-embedding-api Error context can...                           0.1s
 ✘ ui Error                 Head "https://ghcr.io/v2/ru...           0.1s
 ✘ task-executor Error      context canceled                         0.1s
 ✘ api-server Error         context canceled                         0.1s
 ✘ vector-db-api Error      context canceled                         0.1s
Error response from daemon: Head "https://ghcr.io/v2/rubra-ai/rubra/ui/manifests/main": denied: denied
```

Make sure you have logged in with the GitHub Container Registry (ghcr.io) using the following command:

```bash
echo $CR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

Replace `$CR_PAT` with your personal access token (PAT) and `YOUR_GITHUB_USERNAME` with your GitHub username.

2. If you get the following error:

```shell
TVMError: after determining tmp storage requirements for inclusive_scan: cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device
```

This error occurs when the NVIDIA GPU architecture is less than the `sm_80`. Please refer to [this website](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for the supported GPU with architectures `sm_80` and above.

### Credits

```
@software{mlc-llm,
    author = {MLC team},
    title = {{MLC-LLM}},
    url = {https://github.com/mlc-ai/mlc-llm},
    year = {2023}
}
```
