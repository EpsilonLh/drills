# Copyright (c) 2019, SCALE Lab, Brown University. BSD-3-Clause; see LICENSE.
import json
import re
from subprocess import check_output

import numpy as np


def extract_features(design_file, config):
    tools = config['runtime']
    output = check_output([tools['yosys_binary'], '-QT', '-p',
                           f'read_verilog "{design_file}"; stat -json'], text=True)
    start = re.search(r'(?m)^\s*\{', output).end() - 1
    report, _ = json.JSONDecoder().raw_decode(output[start:])
    module = next(iter(report['modules'].values()))
    count, cells = module['num_cells'], module['num_cells_by_type']
    output = check_output([tools['abc_binary'], '-c',
                           f'read_verilog "{design_file}"; print_stats'], text=True)
    inputs, outputs = re.findall(r'i/o\s*=\s*(\d+)\s*/\s*(\d+)', output)[-1]
    features = dict(input_pins=int(inputs), output_pins=int(outputs), cells=count)
    for name, field in [('edges', 'edge'), ('levels', 'lev'), ('latches', 'lat')]:
        features[name] = int(re.findall(r'\b' + field + r'\s*=\s*(\d+)', output)[-1])
    for name, gate in [('and_fraction', '$and'), ('or_fraction', '$or'), ('not_fraction', '$not')]:
        features[name] = cells.get(gate, 0) / max(count, 1)
    return np.asarray([features[name] for name in config['method']['features']], dtype=np.float32)
