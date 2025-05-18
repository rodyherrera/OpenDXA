class LammpstrjParser:
    def __init__(self, filename):
        self._filename= filename
        self._timesteps = []
        self._data = {}

    def parse(self, fields_of_interest=('id', 'x', 'y', 'z')):
        with open(self._filename, 'r') as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            if lines[i].startswith('ITEM: TIMESTEP'):
                timestep = int(lines[i + 1].strip())
                self._timesteps.append(timestep)
                i += 2

                assert lines[i].startswith('ITEM: NUMBER OF ATOMS')
                number_of_atoms = int(lines[i + 1].strip())
                i += 2

                assert lines[i].startswith('ITEM: BOX BOUNDS')
                box = []
                for j in range(3):
                    lo, hi = map(float, lines[i + 1 + j].strip().split())
                    box.append([ lo, hi ])
                i += 4

                assert lines[i].startswith('ITEM: ATOMS')
                header = lines[i].strip().split()[2:]
                indices = { key: header.index(key) for key in fields_of_interest }
                i += 1

                ids = []
                positions = []

                for _ in range(number_of_atoms):
                    parts = lines[i].strip().split()
                    ids.append(int(parts[indices['id']]))
                    position = tuple(float(parts[indices[axis]]) for axis in ('x', 'y', 'z'))
                    positions.append(position)
                    i += 1
                
                self._data[timestep] = {
                    'box': box,
                    'ids': ids,
                    'positions': positions
                }
            else:
                i += 1
        
    def get_timestep_data(self, timestep):
        return self._data.get(timestep, None)

parser = LammpstrjParser('./nanoparticle.lammpstrj')
parser.parse()

print(parser.get_timestep_data(500))