from collections import Counter
from os import path
from typing import Optional

import numpy as np

from aml import Classifier, Validator, utility


class Attack:
    """Attack base class defines common functionality"""

    def __init__(self, name, def_iter, validator, uuid, save, iters,
                 silent, attack_conf):
        self.uuid = uuid
        self.name = name
        self.validator_kind = validator
        self.cls: Optional[Classifier] = None
        self.max_iter = def_iter if iters < 1 else iters
        self.save_records = save
        self.silent = silent
        self.ori_x = None
        self.ori_y = None
        self.adv_x = None
        self.adv_y = None
        self.evasions = None
        self.valid_result = None
        self.validation_reasons = None
        self.reset()
        self.attack_conf = attack_conf or {}

    def reset(self):
        self.cls = None
        self.ori_x = np.array([])
        self.ori_y = np.array([])
        self.adv_x = np.array([])
        self.adv_y = np.array([])
        self.evasions = np.array([])
        self.valid_result = np.array([])
        self.validation_reasons = {}
        return self

    @property
    def n_records(self):
        return len(self.ori_x)

    @property
    def n_evasions(self):
        return len(self.evasions)

    @property
    def n_valid(self):
        return len(self.idx_valid_evades)

    @property
    def use_validator(self):
        return self.validator_kind is not None

    @property
    def idx_valid_evades(self):
        if self.use_validator:
            v_ids = [i for i, s in enumerate(self.valid_result) if s]
            both = list(set(v_ids).intersection(set(self.evasions)))
            return np.array(both)
        return self.evasions.copy()

    @property
    def has_evasions(self):
        return len(self.idx_valid_evades) > 0

    @property
    def has_invalid(self):
        return len(self.validation_reasons.keys()) > 0

    @property
    def error(self):
        target, x_adv = np.array(self.ori_x), np.array(self.adv_x)
        errors = np.linalg.norm((target - x_adv), axis=1)
        errors = errors[self.evasions] if len(self.evasions) else [0]
        return min(errors), max(errors) if self.has_evasions else (0, 0)

    @property
    def init_proto(self):
        return self.get_proto_stats(self.ori_x)

    @property
    def adv_proto(self):
        return self.get_proto_stats(
            self.adv_x[self.evasions] if self.n_evasions > 0 else [])

    @property
    def adv_proto_valid(self):
        input_ = self.adv_x[self.idx_valid_evades] \
            if self.n_valid > 0 else []
        return self.get_proto_stats(input_)

    @property
    def label_stats(self) -> dict:
        labels = []
        if self.use_validator and self.n_valid > 0:
            labels = self.adv_y[
                self.idx_valid_evades].flatten().tolist()
        elif self.n_evasions > 0:
            labels = self.adv_y[self.evasions].flatten().tolist()
        return dict([(self.cls.text_label(c), labels.count(c))
                     for c in self.cls.classes])

    def get_proto_stats(self, records) -> dict:
        if not self.use_validator:
            return {}
        labels = [Validator.determine_proto(
            self.validator_kind, self.cls.attrs, r).name for r in
                  records]
        return dict(Counter(labels))

    def set_cls(self, cls: Classifier, indices=None):
        self.cls = cls
        indices = indices or range(cls.n_test)
        self.ori_x = cls.test_x.copy()[indices, :]
        self.ori_y = cls.test_y.copy()[indices]
        return self

    @staticmethod
    def non_bin_attributes(np_array):
        """Get column indices of non-binary attributes"""
        return [feat for feat in range(len(np_array[0]))
                if len(list(set(np_array[:, feat]))) > 2]

    def run(self):
        pass

    def eval_examples(self):
        """
        Do prediction on adversarial vs original records and determine
        which records succeed at evading expected class label.
        """
        # predictions for original instances
        ori_in = self.cls.formatter(self.ori_x, self.ori_y)
        original = self.cls.predict(ori_in).flatten().tolist()
        correct = np.array((np.where(
            np.array(self.ori_y) == original)[0]).flatten().tolist())

        # adversarial predictions for same data
        adv_in = self.cls.formatter(self.adv_x, self.ori_y)
        adversarial = self.cls.predict(adv_in).flatten().tolist()
        self.adv_y = np.array(adversarial)

        # adversary succeeds iff predictions differ
        evades = np.array(
            (np.where(self.adv_y != original)[0]).flatten().tolist())
        self.evasions = np.intersect1d(evades, correct)

    def validate(self):
        if self.use_validator and self.n_evasions > 0:
            attrs = self.cls.attrs[:self.cls.n_features]
            records = self.cls.denormalize(self.adv_x)
            result, reasons = Validator.validate_records(
                self.validator_kind, attrs, records)
            self.validation_reasons = reasons
            self.valid_result = np.array(result)

    def dump_result(self, out_dir):
        """Write to csv file original and adversarial examples."""
        self.__dump(self.ori_x, self.ori_y, out_dir, 'ori')
        self.__dump(self.adv_x, self.adv_y, out_dir, 'adv')

    def __dump(self, x, y, out_dir, name):
        x = self.cls.denormalize(x.copy())
        attrs = self.cls.attrs[:]
        int_cols = self.cls.mask_cols + [self.cls.n_features]
        labels = y[self.evasions].reshape(-1, 1)
        rows = np.append(x[self.evasions, :], labels, 1)
        fn = f'{self.uuid}_{self.cls.fold_n}_{name}.csv'

        # if validator is used, append validation result
        if self.use_validator and self.n_evasions > 0:
            valid = self.valid_result[self.evasions].reshape(-1, 1)
            int_cols.append(len(rows[0]))
            rows = np.append(rows[:, :], valid, 1)
            attrs.append('valid')

        utility.write_dataset(
            path.join(out_dir, fn), attrs, rows, int_cols)
