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

    @property
    def proto_names(self):
        protos = []
        for record in self.raw_rata:
            for item in record['_Result__proto_init']:
                protos += item.keys()
        return sorted(list(set(protos)))

    @staticmethod
    def extract_values(record):
        fs = record['_Result__f_score']
        acc = record['_Result__accuracy']
        n_rec = mean(record['_Result__n_records'])
        n_ev = mean(record['_Result__n_evasions'])
        n_valid = sum(record['_Result__n_valid'])
        bm = sum([r['benign'] for r in record['_Result__labels']])
        mb = sum([r['malicious'] for r in record['_Result__labels']])
        return [
            record['dataset_name'],
            record['attack'],
            record['robust'],
            record['max_iter'],
            f"{round(mean(fs), 2)} ± {round(stdev(fs), 2)}",
            f"{round(mean(acc), 2)} ± {round(stdev(acc), 2)}",
            round(n_ev / n_rec, 2) if n_rec > 0 else 0,
            round(mean(record['_Result__n_valid']) / n_ev, 2)
            if n_ev > 0 else 0,
            f"{(100 * bm / n_valid) if n_valid > 0 else 0:.0f}/"
            f"{(100 * mb / n_valid) if n_valid > 0 else 0:.0f}",
        ]

    @staticmethod
    def proto_freq(record, labels, init_key, succ_key):
        return [round(p, 2) for p in [
            mean([succ[lbl] / init[lbl]
                  if lbl in init and init[lbl] > 0 and lbl in succ
                  else 0 for init, succ in
                  zip(record[init_key],
                      record[succ_key])])
            for idx, lbl in enumerate(labels)]]

    @staticmethod
    def extract_proto_values(record, labels):
        return [record['dataset_name'],
                record['attack'],
                record['robust'],
                record['max_iter']] + \
               Results.proto_freq(
                   record, labels, '_Result__proto_init',
                   '_Result__proto_evasions') + \
               Results.proto_freq(
                   record, labels, '_Result__proto_evasions',
                   '_Result__proto_valid')

    def exp_table(self):
        headers = ["#", "Dataset", "Attack", "Robust", "Iters",
                   "F-score", "Accuracy", "Evasions", "Valid", "B/M"]
        mat = [self.extract_values(record) for record in self.raw_rata]
        return headers, mat

    def proto_table(self):
        headers = ["#", "Dataset", "Attack", "Robust", "Iters"] + \
                  [f"E-{p}" for p in self.proto_names] + \
                  [f"V-{p}" for p in self.proto_names]
        mat = [self.extract_proto_values(record, self.proto_names)
               for record in self.raw_rata]
        return headers, mat

    def reasons_table(self):
        headers = ["#", "Dataset", "Attack", "Proto", "Reason", "Freq"]
        result = {}
        for record in self.raw_rata:
            ds, att = record['dataset_name'], record['attack']
            if ds not in result:
                result[ds] = {}
            if att not in result[ds]:
                result[ds][att] = {}
            for p in self.proto_names:
                if p not in result[ds][att]:
                    result[ds][att][p] = {}
                for folds in record['_Result__validations']:
                    for k, v in folds.items():
                        if str(k).startswith(p):
                            if k not in result[ds][att][p]:
                                result[ds][att][p][k] = v
                            else:
                                result[ds][att][p][k] += v
        mat = []
        for ds, atts in result.items():
            for att, protos in atts.items():
                for proto, ress in protos.items():
                    reasons = sorted(
                        list(ress.items()), reverse=True,
                        key=lambda x: x[1])
                    tot = sum([r[1] for r in reasons])
                    for r, v in reasons:
                        r, v = r.replace(proto, '', 1), v / tot
                        mat.append([ds, att, proto, r, round(v, 2)])
        return headers, mat

    def write_table(self, headers, mat, file_name, sorter=None):
        file_ext = 'txt' if self.format != 'tex' else 'tex'
        fn = os.path.join(self.directory, f'{file_name}.{file_ext}')
        writer = SpaceAlignedTableWriter() if self.format != 'tex' \
            else LatexTableWriter()
        mat_sort = sorter or (lambda x: (x[0], x[1], x[2], x[3]))
        mat = sorted(mat, key=mat_sort)

        for n, r in enumerate(mat):
            mat[n] = [n + 1] + r
        writer.headers = headers
        writer.value_matrix = mat
        writer.write_table()
        writer.dump(fn)
        logger.debug(f'Saved to {fn}')


def plot_results(directory, fmt):
    res = Results(directory, fmt)
    if res.n_results == 0:
        logger.warning("No results found in results directory.")
        logger.warning("Nothing was plotted.")
        return
    res.write_table(*res.exp_table(), 'table')
    res.write_table(*res.proto_table(), 'table_proto')
    res.write_table(*res.reasons_table(), 'table_reasons',
                    sorter=lambda x: (x[0], x[1], x[2], -x[4]))
