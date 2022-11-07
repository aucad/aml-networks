import json
import glob
import os
from statistics import mean

from pytablewriter import SpaceAlignedTableWriter, MarkdownTableWriter


class Results:

    def __init__(self, directory):
        self.raw_rata = []
        self.robust = set()
        self.datasets = set()
        self.attacks = set()
        self.iters = set()
        self.directory = directory

    def add(self, result):
        self.raw_rata.append(result)
        self.datasets.add(result['dataset_name'])
        self.robust.add(result['robust'])
        if 'attack' in result:
            self.attacks.add(result['attack'])
            self.iters.add(result['max_iter'])

    def load_results(self):
        json_files = glob.glob(os.path.join(self.directory, "*.json"))
        for file in json_files:
            data = json.load(open(file))
            self.add(data)

    def extract_values(self, record):
        n_rec = mean(record['_Result__n_records'])
        n_ev = mean(record['_Result__n_evasions'])
        return [
            record['dataset_name'],
            record['attack'],
            record['max_iter'],
            record['robust'],
            round(mean(record['_Result__f_score']), 2),
            round(n_ev / n_rec, 2) if n_rec > 0 else 0,
            round(mean(record['_Result__n_valid']) / n_ev, 2)
            if n_ev > 0 else 0
        ]

    def write_table(self):
        writer = SpaceAlignedTableWriter()
        writer.headers = ["#",
                          "Dataset", "Attack", "Iters", "Robust",
                          "F-score", "Evasions", "Valid"]
        mat = []
        for record in self.raw_rata:
            mat.append(self.extract_values(record))
        mat = sorted(mat, key=lambda x: (x[0], x[1], x[2], x[3]))
        for n, r in enumerate(mat):
            mat[n] = [n + 1] + r
        writer.value_matrix = mat
        writer.write_table()
        writer.dump(os.path.join(self.directory, 'table.txt'))


if __name__ == '__main__':
    res = Results('output-2')
    res.load_results()
    res.write_table()
