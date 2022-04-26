import json
import site
import os
from pathlib import Path

import typer
import onnx
from jinja2 import Template
from cookiecutter.main import cookiecutter
from google.protobuf.json_format import MessageToJson

from smartonnx.utils import (
    build_graph_template,
    build_tensor_template,
    get_graph_inputs,
    get_graph_tensors,
)


app = typer.Typer(help="smartonnx - Convert ONNX Models to Cairo Contracts")


@app.command()
def convert(
    model_path: Path = typer.Argument(..., help="Path to the ONNX model"),
    cairo_package: Path = typer.Argument(..., help="Path to the ONNX model"),
):

    """
    Convert a ONNX model to a Cairo package

    Args:
        model (Path): Path to the ONNX model
    """

    # TODO: check the model is valid
    onnx_model = onnx.load(model_path)
    s = MessageToJson(onnx_model)
    onnx_json = json.loads(s)

    model_def_path = Path("./onnx_def.json")
    with open(model_def_path.resolve(), "w") as f:
        json.dump(onnx_json, f)
    # print(json.dumps(onnx_json, sort_keys=True, indent=4))

    site_packages = site.getsitepackages()[0]
    graph_contract_path = Path(
        os.path.join(site_packages, "smartonnx/templates/contract.cairo.tmpl")
    )

    cookiecutter(
        "https://github.com/franalgaba/felucca-package-template.git",
        extra_context={"project_name": cairo_package.name},
        no_input=True,
        overwrite_if_exists=True,
        checkout="feature/onnx",
    )

    cairo_package = Path(cairo_package) / cairo_package.name.replace("-", "_")
    build_graph_template(graph_contract_path, cairo_package / "contract.cairo")

    typer.echo("Building ONNX tensors contracts")
    # for tensor in get_graph_tensors():
    #     tensor_contract_path = Path(
    #         os.path.join(site_packages, "smartonnx/templates/tensor_loader.cairo.tmpl")
    #     )
    #     build_tensor_template(
    #         tensor_contract_path,
    #         cairo_package / f"{tensor.name}_tensor_loader.cairo",
    #     )

    os.remove(model_def_path.resolve())


@app.command()
def greet():
    pass


if __name__ == "__main__":
    app()
