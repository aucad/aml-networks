"""
Utility for generating plots of the captured results.
"""

import glob
import json
import logging
import os
from statistics import mean, stdev

# noinspection PyPackageRequirements
import pandas as pd
from pytablewriter import SpaceAlignedTableWriter, LatexTableWriter

from src import sdiv
from src.utility import smean, rarr, rget

logger = logging.getLogger(__name__)


class ResultsPlot:

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
        if 'iters' in result:
            self.iters.add(result['max_iter'])

    def load_results(self):
        json_files = glob.glob(os.path.join(self.directory, "*.json"))
        for file in json_files:
            data = json.load(open(file))
            self.add(data)
        return self

    @staticmethod
    def std_cols(record):
        """For all tables"""
        return [rget(record, 'classifier'),
                rget(record, 'dataset_name'),
                'T' if rget(record, 'robust') else 'F',
                rget(record, 'attack'),
                rget(record, 'max_iter')]

    @property
    def std_hd(self):
        return ["CLS", "Dataset", "R", "Attack", "i"]

    @property
    def proto_names(self):
        protos = []
        for record in self.raw_rata:
            for item in record[rarr('proto_init')]:
                protos += item.keys()
        return sorted(list(set(protos)))

    @staticmethod
    def proto_freq(record, labels, init_key, success_key):
        """Average success rate by protocol"""
        return [round(p, 2) for p in [smean(
            [sdiv(rget(b, lbl, 0), rget(a, lbl, 0))
             for a, b in zip(record[init_key], record[success_key])])
            for lbl in labels]]

    def evasion_table(self):
        def extract_values(record):
            lbl = record[rarr('labels')]
            vld = record[rarr('n_valid')]
            fs = record[rarr('f_score')]
            ac = record[rarr('accuracy')]
            nr = smean(record[rarr('n_records')])
            ne = smean(record[rarr('n_evasions')])
            nv = sum(vld) if ne > 0 else 0
            bm = sum([r['benign'] for r in lbl])
            mb = sum([r['malicious'] for r in lbl])
            return ResultsPlot.std_cols(record) + [
                f"{round(mean(fs), 2)} ± {round(stdev(fs), 2)}",
                f"{round(mean(ac), 2)} ± {round(stdev(ac), 2)}",
                round(sdiv(ne, nr), 2),
                round(sdiv(smean(vld), ne), 2),
                f"{100 * sdiv(bm, nv):.0f}/{100 * sdiv(mb, nv):.0f}"]

        h = ["F-score", "Accuracy", "Evades", "Valid", "B / M"]
        mat = [extract_values(record) for record in self.raw_rata]
        return self.std_hd + h, mat

    def proto_table(self):
        proto1 = rarr('proto_init'), rarr('proto_evasions')
        proto2 = rarr('proto_evasions'), rarr('proto_valid')

        def extract_proto_values(record, labels):
            return (ResultsPlot.std_cols(record) +
                    ResultsPlot.proto_freq(record, labels, *proto1) +
                    ResultsPlot.proto_freq(record, labels, *proto2))

        h = [f"e/{p}" for p in self.proto_names] + \
            [f"v/{p}" for p in self.proto_names]
        mat = [extract_proto_values(record, self.proto_names)
               for record in self.raw_rata]
        return self.std_hd + h, mat

    def reasons_table(self):
        headers = ["Dataset", "Attack", "Proto", "Reason", "Freq"]
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
                for folds in record[rarr('validations')]:
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
        try:
            mat = sorted(mat, key=sorter) if sorter else mat
        except TypeError:
            # sort can fail if nulls in data
            pass

        headers = ["#"] + headers
        for n, r in enumerate(mat):
            mat[n] = [n + 1] + r
        writer.headers = headers
        writer.value_matrix = mat
        writer.write_table()
        writer.dump(fn)
        logger.debug(f'Saved to {fn}')

    def show_duration(self):
        div = 72 * "="
        ts = pd.to_timedelta(sum(
            [r['end'] - r['start'] for r in self.raw_rata]))
        print(f'{div}\nExperiment duration: {ts}\n{div}')


def plot_results(directory, fmt):
    res = ResultsPlot(directory, fmt)
    if res.n_results == 0:
        logger.warning("No results found in results directory.")
        logger.warning("Nothing was plotted.")
        return
    res.write_table(
        *res.evasion_table(), 'table',
        sorter=lambda x: (x[0], x[1], x[3], x[2], x[4]))
    res.write_table(
        *res.proto_table(), 'table_proto',
        sorter=lambda x: (x[0], x[1], x[3], x[2], x[4]))
    res.write_table(
        *res.reasons_table(), 'table_reasons',
        sorter=lambda x: (x[0], x[1], x[2], -x[4]))
    res.show_duration()
