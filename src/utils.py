import os
from langgraph.graph import Graph
from PIL import Image

def graph_to_image(graph: Graph, file_name: str = "image.png") -> Image:
    """
    Converts a LangGraph graph to an image and saves it to img path
    Args:
        graph (Graph): The LangGraph graph to convert.
        file_name (str): The path where the image will be saved.
    """
    image_path = f"../img/{file_name}"

    if os.path.exists(image_path):
        return f"File {file_name} already exists. Skipping image generation."
        
    # Generate a PNG image from the graph
    else:
        graph.get_graph().draw_mermaid_png(output_file_path=image_path)
        print(f"Graph image saved to {image_path}")
        return Image.open(image_path)