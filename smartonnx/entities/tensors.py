class Tensor:
    dims = None
    float_data = None
    name = ""
    output = None

    def __init__(self, dims, float_data, output, name) -> None:
        self.dims = dims
        self.float_data = float_data
        self.output = output
        self.name = name
