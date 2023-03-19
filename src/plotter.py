"""
Utility for generating plots of the captured results.
"""

import glob
import json
import logging
import os
from statistics import mean, stdev

# noinspection PyPackageRequirements
import numpy as np
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
                '✓' if rget(record, 'robust') else ' ',
                rget(record, 'dataset_name'),
                rget(record, 'attack'),
                rget(record, 'max_iter')]

    @property
    def std_hd(self):
        return ["CLS", "R", "DS", "ATK", "i"]

    @property
    def proto_names(self):
        protos = []
        for record in self.raw_rata:
            for item in record[rarr('proto_init')]:
                protos += item.keys()
        return sorted(list(set(protos).intersection(
            {'tcp', 'udp', 'oth'})))

    @staticmethod
    def proto_freq(record, labels, init_key, success_key, ev_key):
        """Average success rate by protocol"""
        total = sum(record[init_key])
        rec = smean(record[rarr(ev_key)])
        return [round(sdiv(sum([
            rget(b, lbl, 0) for b in record[success_key]]), total), 2)
                if round(sdiv(total, rec), 2) > 0 else 0
                for lbl in labels]

    def classifier_table(self):

        def row_values(rec):
            return [
                rget(rec, 'dataset_name'),
                rget(rec, 'classifier'),
                rget(rec, 'robust'),
                rec[rarr('f_score')],
                rec[rarr('accuracy')]]

        def match_rows(tbl, ds, cl, rb):
            temp = tbl[tbl[:, 0] == ds, :]
            temp = temp[temp[:, 1] == cl, :]
            return temp[temp[:, 2] == rb, :]

        def collapse(ds, cl, rb, rows):
            fs = np.array([np.array(f) for f in rows[:, 3]]) \
                .flatten().tolist()
            ac = np.array([np.array(f) for f in rows[:, 4]]) \
                .flatten().tolist()
            # noinspection PyTypeChecker
            return [ds, cl, rb,
                    f"{round(mean(fs), 2)} ± {round(stdev(fs), 2)}",
                    f"{round(mean(ac), 2)} ± {round(stdev(ac), 2)}"]

        h = ["DS", "CLS", "R", "F-score", "Accuracy"]
        tb = np.array([row_values(record) for record in self.raw_rata])
        mat = [collapse(d, c, r, match_rows(tb, d, c, r))
               for d in np.unique(tb[:, 0])
               for c in np.unique(tb[:, 1])
               for r in np.unique(tb[:, 2])]
        return h, mat

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
            av_ev = round(sdiv(ne, nr), 2)
            bl = sdiv(bm, nv)
            return ResultsPlot.std_cols(record) + [
                f"{round(mean(fs), 2)} ± {round(stdev(fs), 2)}",
                f"{round(mean(ac), 2)} ± {round(stdev(ac), 2)}",
                av_ev,
                round(sdiv(smean(vld), ne), 2) if av_ev > 0 else 0,
                f"{100 * bl:.0f}--{100 * (1. - bl):.0f}"
                if av_ev > 0 else 0]

        h = ["F-score", "Accuracy", "Evade", "Valid", "B / M"]
        mat = [extract_values(record) for record in self.raw_rata]
        return self.std_hd + h, mat

    def proto_table(self):
        proto1 = rarr('n_evasions'), rarr('proto_evasions'), 'n_records'
        proto2 = rarr('n_valid'), rarr('proto_valid'), 'n_evasions'

        def extract_proto_values(record, labels):
            evades = ResultsPlot.proto_freq(record, labels, *proto1)
            valid = [v if e > 0 else 0 for e, v in zip(
                evades, ResultsPlot.proto_freq(
                    record, labels, *proto2))]
            return ResultsPlot.std_cols(record) + evades + valid

        h = [f"e/{p}" for p in self.proto_names] + \
            [f"v/{p}" for p in self.proto_names]
        mat = [extract_proto_values(record, self.proto_names)
               for record in self.raw_rata]
        return self.std_hd + h, mat

    def reasons_table(self):
        headers = ["DS", "ATK", "Proto", "Reason", "Freq"]
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
        *res.classifier_table(), 'table_cls',
        sorter=lambda x: (x[0], x[1], x[2]))
    res.write_table(
        *res.evasion_table(), 'table_evades',
        sorter=lambda x: (x[0], x[2], x[3], x[1], x[4]))
    res.write_table(
        *res.proto_table(), 'table_proto',
        sorter=lambda x: (x[0], x[2], x[3], x[1], x[4]))
    res.write_table(
        *res.reasons_table(), 'table_reasons',
        sorter=lambda x: (x[0], x[1], x[2], -x[4]))
    res.show_duration()
