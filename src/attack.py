from typing import Optional

from cls import AbsClassifierInstance


class AbsAttack:

    def __init__(self, name):
        self.name = name
        self.cls: Optional[AbsClassifierInstance] = None

    def set_cls(self, cls):
        self.cls = cls
        return self

    def run(self):
        pass

    @staticmethod
    def default_run(attack_cls):
        from utility import DEFAULT_DS
        from loader import ClsLoader
        # load default classifier
        c = ClsLoader.load().load(DEFAULT_DS, 0.995).train()
        # run specified attack on default classifier
        attack_cls().set_cls(c).run()
