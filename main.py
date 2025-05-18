from core.lammpstrj_parser import LammpstrjParser
from core.hybrid_neighbor_finder import HybridNeighborFinder

parser = LammpstrjParser('./nanoparticle.lammpstrj')

for data in parser.iter_timesteps():
    timestep = data['timestep']
    positions = data['positions']
    box = data['box']
    ids = data['ids']

    print(f'Timestep {timestep}, {len(positions)} atoms')

    neighbor_finder = HybridNeighborFinder(
        positions=positions,
        cutoff=3.5,
        num_neighbors=12,
        box_bounds=box
    )

    print('Neighbors...')
    neighbors = neighbor_finder.find_neighbors()

    print(f'Neighbors of atom {ids[0]}: {neighbors[0]}')
    break
