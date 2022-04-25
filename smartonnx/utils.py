import os
import json
import subprocess
from pathlib import Path

import typer
from smartonnx.entities.inputs import Input
from smartonnx.entities.operators import Operator

from smartonnx.entities.tensors import Tensor


def execute_subprocess(command: str, **kwargs) -> None:
    """
    Execute a subprocess given a command

    Args:
        command (str): command to execute
        kwargs: extra args for subprocess

    Raises:
        Exit: if the command fails
    """

    try:
        subprocess.run(
            command, shell=True, universal_newlines=True, check=True, **kwargs
        )
    except subprocess.CalledProcessError:
        raise typer.Exit(code=1)


def execute_poetry(command: str) -> None:
    """
    Execute a poetry command with subprocess

    Args:
        command (str): command to execute
    """

    poetry = "$HOME/.poetry/bin/poetry"
    if "POETRY_HOME" in os.environ:
        poetry = "$POETRY_HOME/bin/poetry"
    execute_subprocess(f"{poetry} {command}")


def _load_onnx_def():
    path = Path("./onnx_def.json")
    with open(str(path.resolve())) as f:
        onnx_def = json.load(f)
    return onnx_def


def get_graph_name():
    onnx_def = _load_onnx_def()
    return onnx_def["graph"]["name"]


def get_graph_tensors():
    tensor_nodes = []
    onnx_def = _load_onnx_def()
    for node in onnx_def["graph"]["node"]:
        if "attribute" in node:
            if node["attribute"][0]["type"] == "TENSOR":
                tensor_nodes.append(
                    Tensor(
                        node["attribute"][0]["t"]["dims"],
                        node["attribute"][0]["t"]["floatData"],
                        node["output"],
                        node["name"],
                    )
                )
    return tensor_nodes


def get_graph_inputs():
    input_list = []
    onnx_def = _load_onnx_def()
    if "input" in onnx_def["graph"]:
        input = onnx_def["graph"]["input"][0]
        input_list.append(
            Input(
                input["type"]["tensorType"]["shape"]["dim"][0]["dimValue"],
                input["name"],
            )
        )
    return input_list


def get_graph_operators():
    operator_nodes = []
    onnx_def = _load_onnx_def()
    for node in onnx_def["graph"]["node"]:
        if "type" not in node:
            operator_nodes.append(
                Operator(node["input"], node["output"], node["opType"])
            )
