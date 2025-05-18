from core.lammpstrj_parser import LammpstrjParser

parser = LammpstrjParser('./nanoparticle.lammpstrj')

for data in parser.iter_timesteps():
    timestep = data['timestep']
    positions = data['positions']
    box = data['box']
    ids = data['ids']

    print(f'Timestep {timestep}, {len(positions)} atoms')
    break
