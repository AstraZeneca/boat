"""A template pipeline ."""

from kfp import compiler, dsl, kubernetes
from kfp.kubernetes import use_secret_as_env

IMAGE_TAG = "dev"
MOUNT_VOLUME = "shared-vol"
MOUNT_PATH = "/vol-1"
PROXY_URL = "http://azpse.astrazeneca.net:9480"
NO_PROXY = (
    "10.0.0.0/8,172.29.0.0/8,astrazeneca.net,localhost,127.0.0.1,::1,.kubeflow,"
    ".snowflakecomputing.com,169.254.169.254"
    )

# ------------------------------------------- Pod settings ----------------------------------------#
instances = {
    "m5.xlarge": {"cpu": "2", "memory": "10G"},
    "m5.2xlarge": {"cpu": "6", "memory": "24G"},
    "m5.4xlarge": {"cpu": "10", "memory": "54G"},
    "m5.8xlarge": {"cpu": "28", "memory": "116G"},
    "g5.xlarge": {"cpu": "2", "memory": "10G"},
    "g5.8xlarge": {"cpu": "27", "memory": "110G"},
}


def set_resources(task, instance="m5.xlarge"):
    """Set resources for the task."""
    resourcing = instances.get(instance, {"cpu": "2", "memory": "10G"})
    task.set_cpu_limit(resourcing["cpu"]).set_memory_limit(resourcing["memory"])
    kubernetes.add_node_selector(task=task, label_key="instance-type", label_value=instance)
    instance = f"gpu_{instance}" if instance.startswith("g") else instance
    kubernetes.add_toleration(
        task, key=f"as_{instance.replace('.', '_')}_ns", operator="Equal", value="true", effect="NoSchedule"
    )
    kubernetes.add_toleration(
        task, key=f"as_{instance.replace('.', '_')}_ne", operator="Equal", value="true", effect="NoExecute"
    )


def set_config(task):
    """Add setup configs to the tasks."""
    # Setup the proxy
    for v in ["http_proxy", "https_proxy"]:
        task.set_env_variable(v, PROXY_URL).set_env_variable(v.upper(), PROXY_URL)
    task.set_env_variable("no_proxy", NO_PROXY).set_env_variable("no_proxy".upper(), NO_PROXY)
        
    # Setting Image pull policy to always pick latest
    kubernetes.set_image_pull_policy(task=task, policy="Always")

    # Setup W&B
    use_secret_as_env(
        task,
        secret_name="mlab-wandb",
        secret_key_to_env={
            "WANDB_BASE_URL": "WANDB_BASE_URL",
            "WANDB_API_KEY": "WANDB_API_KEY",
        },
    )

    # mount shared volume
    kubernetes.mount_pvc(
        task,
        pvc_name=MOUNT_VOLUME,
        mount_path=MOUNT_PATH,
    )   


# -------------------------------------------  Components------------------------------------------#
@dsl.container_component
def first_component(**kwargs):
    """Load and parse configuration, then run annotation on the input sequence."""
    return dsl.ContainerSpec(
        image=f"harbor.csis.astrazeneca.net/your-project/your-first-component:{IMAGE_TAG}",
        command=["python", "src/main.py"],
        args=[a for k, v in kwargs.items() for a in (f"--{k}", v)],
    )


@dsl.container_component
def second_component(**kwargs):
    """Load and parse configuration, then run annotation on the input sequence."""
    return dsl.ContainerSpec(
        image=f"harbor.csis.astrazeneca.net/your-project/your-second-component:{IMAGE_TAG}",
        command=["python", "src/main.py"],
        args=[a for k, v in kwargs.items() for a in (f"--{k}", v)],
    )


# -------------------------------------------  Pipeline   -------------------------------------------------------------#
@dsl.pipeline(
    name="Template Pipeline",
    description="This is just a template sample placeholder.",
)
def template_pipeline(**kwargs):
    """
    Build pipeline.

    All input parameters are passed through Kubeflow GUI and then passed to the components that need them
    """
    task_1 = first_component(**{k: kwargs[k] for k in ["an_arg_first_component", "another_arg_first_component"]})
    set_config(task_1)
    set_resources(task_1, instance="m5.xlarge")
    
    task_2 = second_component(**{k: kwargs[k] for k in ["an_arg_second_component", "another_arg_second_component"]})
    set_config(task_2)
    set_resources(task_2, instance="m5.xlarge")

    task_2.after(task_1)


compiler.Compiler().compile(template_pipeline, "template_pipeline.yaml")
