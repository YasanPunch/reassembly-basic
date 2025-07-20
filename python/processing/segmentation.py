import numpy as np


class Segmentation:
    def __init__(self, processing_panel):
        self._processing_panel = processing_panel

    def region_growing_mesh(self, m):
        m.compute_vertex_normals(True)
        m.compute_adjacency_list()

        def angle_rad_between_principle(n):
            dot_product = np.dot(n, [0, 0, 0])
            # Clip for numerical stability
            return np.arccos(np.clip(dot_product, -1.0, 1.0))

        def is_similar(n1, n2):
            dot_product = np.dot(n1, n2)
            # Clip for numerical stability
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            threshold = 0.1
            if angle_rad < threshold and angle_rad > -threshold:
                return True
            else:
                return False

        # key angle between principle
        # value indices array
        regions = dict()

        for n1 in np.asarray(m.vertex_normals):
            a = angle_rad_between_principle(n1)
            added = False
            for abp in regions:
                if is_similar(a, abp):
                    regions[abp].append(a)
                    added = True
            if not added:
                regions[a] = [a]

        print(len(regions))
