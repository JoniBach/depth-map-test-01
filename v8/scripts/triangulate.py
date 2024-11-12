import json
import numpy as np
from scipy.spatial import Delaunay

def triangulate_quadrilaterals(triangles, vertices):
    quads = []
    for simplex in triangles:
        v1, v2, v3 = vertices[simplex]

        # Create a quadrilateral by adding midpoints as an example
        quad_vertices = [
            v1,
            v2,
            (v1 + v3) / 2,
            (v2 + v3) / 2
        ]
        quads.append(quad_vertices)
    return quads

def main():
    try:
        # Load data from JSON file
        with open('../TRIANGULATION.json') as f:
            data = json.load(f)

        # Check if data length is even for 2D reshaping (pairs)
        if len(data) % 2 != 0:
            print("The data length is not suitable for 2D coordinates.")
            return

        # Reshape data into 2D coordinates
        vertices = np.array(data).reshape(-1, 2)

        # Perform Delaunay triangulation on 2D points
        triangulation = Delaunay(vertices)
        triangles = triangulation.simplices

        # Generate quadrilaterals
        quads = triangulate_quadrilaterals(triangles, vertices)
        print("Generated quadrilaterals:", quads)
        return quads

    except FileNotFoundError:
        print("The file 'TRIANGULATION.json' was not found.")
    except json.JSONDecodeError as e:
        print(f"An error occurred while parsing the JSON file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
