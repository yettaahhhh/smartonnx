class Operator:
    input_nodes = None
    output_nodes = None
    name = ""

    def __init__(self, input_nodes, output_nodes, name) -> None:
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.name = name
