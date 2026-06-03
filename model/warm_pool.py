"""
WarmPool — состояние «тёплых» функций на каждом узле. Это и есть serverless-ядро.

На узле j хранится словарь {тип_функции: возраст_в_слотах}. Возраст 0 = только что
вызвана. Каждый слот:
  1) все тёплые функции стареют на 1;
  2) использованные в этом слоте освежаются (возраст -> 0);
  3) функции старше keep_alive выселяются (перестали быть тёплыми);
  4) если тёплых больше w_max (лимит памяти узла) — выселяем самые старые (LRU).

Это изолированный конечный автомат → тестируется отдельно (keep-alive, LRU).
"""


class WarmPool:
    def __init__(self, M: int, keep_alive: int, w_max: int):
        self.M = M
        self.keep_alive = keep_alive
        self.w_max = w_max
        # warm[j] = {gtype: age}
        self.warm = {j: {} for j in range(M)}

    def is_warm(self, j: int, g: int) -> bool:
        """Тёплая ли функция g на узле j прямо сейчас."""
        return g in self.warm[j]

    def update(self, used_per_node: dict) -> None:
        """
        Обновить пулы за один тайм-слот.
        used_per_node: {j: set(типов функций, выполненных на узле j в этом слоте)}.
        """
        for j in range(self.M):
            # 1) состарить всё
            for g in list(self.warm[j].keys()):
                self.warm[j][g] += 1
            # 2) освежить использованные
            for g in used_per_node.get(j, ()):
                self.warm[j][g] = 0
            # 3) выселить устаревшие (keep-alive истёк)
            self.warm[j] = {g: a for g, a in self.warm[j].items()
                            if a <= self.keep_alive}
            # 4) LRU при переполнении памяти узла
            if len(self.warm[j]) > self.w_max:
                freshest = sorted(self.warm[j].items(), key=lambda kv: kv[1])[:self.w_max]
                self.warm[j] = dict(freshest)

    def snapshot(self) -> dict:
        """Срез текущего состояния: {j: set(тёплых типов)}. Удобно для evaluate()."""
        return {j: set(d.keys()) for j, d in self.warm.items()}
