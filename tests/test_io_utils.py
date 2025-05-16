import unittest
import os
import shutil
import open3d as o3d
from src.io_utils import load_fragment, load_fragments_from_directory, save_mesh, combine_meshes

class TestIOUtils(unittest.TestCase):

    def setUp(self):
        self.test_data_dir = "temp_test_io_data"
        self.input_dir = os.path.join(self.test_data_dir, "input")
        self.output_dir = os.path.join(self.test_data_dir, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a dummy OBJ file
        self.dummy_obj_path = os.path.join(self.input_dir, "dummy.obj")
        with open(self.dummy_obj_path, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    def tearDown(self):
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def test_load_single_fragment(self):
        mesh = load_fragment(self.dummy_obj_path)
        self.assertIsNotNone(mesh)
        self.assertTrue(mesh.has_vertices())
        self.assertTrue(mesh.has_triangles())

    def test_load_fragments_from_directory(self):
        fragments_data = load_fragments_from_directory(self.input_dir)
        self.assertEqual(len(fragments_data), 1)
        self.assertEqual(fragments_data[0]['name'], "dummy.obj")
        self.assertTrue(fragments_data[0]['mesh'].has_vertices())

    def test_save_and_load_mesh(self):
        mesh = load_fragment(self.dummy_obj_path)
        save_path = os.path.join(self.output_dir, "saved_dummy.obj")
        save_mesh(mesh, save_path)
        self.assertTrue(os.path.exists(save_path))
        
        loaded_saved_mesh = load_fragment(save_path)
        self.assertIsNotNone(loaded_saved_mesh)
        self.assertEqual(len(mesh.vertices), len(loaded_saved_mesh.vertices))
        self.assertEqual(len(mesh.triangles), len(loaded_saved_mesh.triangles))
        
    def test_combine_meshes(self):
        mesh1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        mesh2 = o3d.geometry.TriangleMesh.create_box(extent=(0.1,0.1,0.1))
        mesh2.translate([0.2,0,0])
        
        combined = combine_meshes([mesh1, mesh2])
        self.assertTrue(combined.has_vertices())
        expected_verts = len(mesh1.vertices) + len(mesh2.vertices)
        expected_tris = len(mesh1.triangles) + len(mesh2.triangles)
        self.assertEqual(len(combined.vertices), expected_verts)
        self.assertEqual(len(combined.triangles), expected_tris)


if __name__ == '__main__':
    unittest.main()
