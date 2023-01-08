import json
import glob
import logging
import os
from statistics import mean, stdev

from pytablewriter import SpaceAlignedTableWriter, LatexTableWriter

logger = logging.getLogger(__name__)


class Results:

    def __init__(self, directory, fmt):
        self.raw_rata = []
        self.robust = set()
        self.datasets = set()
        self.attacks = set()
        self.iters = set()
        self.directory = directory
        self.format = fmt
        self.load_results()

    @property
    def n_results(self):
        return len(self.raw_rata)

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
        return self

    @staticmethod
    def extract_values(record):
        fs = record['_Result__f_score']
        acc = record['_Result__accuracy']
        n_rec = mean(record['_Result__n_records'])
        n_ev = mean(record['_Result__n_evasions'])
        return [
            record['dataset_name'],
            record['attack'],
            record['robust'],
            record['max_iter'],
            f"{round(mean(fs), 2)} ± {round(stdev(fs), 2)}",
            f"{round(mean(acc), 2)} ± {round(stdev(acc), 2)}",
            round(n_ev / n_rec, 2) if n_rec > 0 else 0,
            round(mean(record['_Result__n_valid']) / n_ev, 2)
            if n_ev > 0 else 0
        ]

    def write_table(self):
        if self.n_results == 0:
            logger.warning("No results found in results directory.")
            logger.warning("Nothing was plotted.")
            return

        file_ext = 'txt' if self.format != 'tex' else 'tex'
        fn = os.path.join(self.directory, f'table.{file_ext}')
        writer = SpaceAlignedTableWriter() if self.format != 'tex' \
            else LatexTableWriter()
        writer.headers = ["#",
                          "Dataset", "Attack", "Robust", "Iters",
                          "F-score", "Accuracy", "Evasions", "Valid"]
        mat = []
        for record in self.raw_rata:
            mat.append(self.extract_values(record))
        mat = sorted(mat, key=lambda x: (x[0], x[1], x[2], x[3]))
        for n, r in enumerate(mat):
            mat[n] = [n + 1] + r
        writer.value_matrix = mat
        writer.write_table()
        writer.dump(fn)
        logger.debug(f'Saved to {fn}')


def plot_results(directory, fmt):
    res = Results(directory, fmt)
    res.write_table()
