from typing import Optional

from cls import AbsClassifierInstance


class AbsAttack:

    def __init__(self, name):
        self.name = name
        self.cls: Optional[AbsClassifierInstance] = None
        self.out_dir = None

    def set_cls(self, cls: AbsClassifierInstance):
        self.cls = cls
        self.out_dir = cls.out_dir
        return self

    def run(self):
        pass
