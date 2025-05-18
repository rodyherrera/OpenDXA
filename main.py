from core.lammpstrj_parser import LammpstrjParser
from core.hybrid_neighbor_finder import HybridNeighborFinder
from core.ptm_local_classifier import PTMLocalClassifier, get_ptm_templates
import numpy as np

parser = LammpstrjParser('./nanoparticle.lammpstrj')

for data in parser.iter_timesteps():
    timestep = data['timestep']
    positions = data['positions']
    box_bounds = data['box']
    ids = data['ids']

    print(f'Timestep {timestep}, {len(positions)} atoms')

    neighbor_finder = HybridNeighborFinder(
        positions=positions,
        cutoff=3.5,
        num_neighbors=12,
        box_bounds=box_bounds
    )
    print('Finding neighbors...')
    neighbors = neighbor_finder.find_neighbors()
    print(f'Neighbors of atom {ids[0]}:', neighbors[0])

    templates, template_sizes = get_ptm_templates()
    classifier = PTMLocalClassifier(
        positions=positions,
        box_bounds=box_bounds,
        neighbor_dict=neighbors,
        templates=templates,
        template_sizes=template_sizes,
        max_neighbors=template_sizes.max()
    )

    types, quaternions = classifier.classify()
    print('Local structure types:', types[:10])
    print('Orientation quaternions:', quaternions[:10])

    break
