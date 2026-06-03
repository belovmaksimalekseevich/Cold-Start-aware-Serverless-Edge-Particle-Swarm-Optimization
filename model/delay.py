"""
Формулы задержки — ЧИСТЫЕ функции (детерминированы при заданных входах).

Соответствуют формулам диплома 3.1–3.5, плюс НОВЫЙ член — холодный старт.
Всё в секундах (СИ). Чистота => тривиально тестируются числами руками.

  T_trans = D·8 / R                       (3.1) передача данных по каналу
  T_comp  = C / F                         (3.2) вычисление на узле
  T_queue = ρ/(1-ρ) · mean_service        (3.3) ожидание в очереди M/M/1
  T_cold  = T_cold · [функция холодная]   (НОВОЕ) штраф холодного старта
  T_off   = T_trans + T_comp + T_queue + T_cold   (3.4) суммарно при выгрузке
  T_local = C / F_loc                     (3.5) локальное исполнение на устройстве
"""


def t_transmission(D: float, R: float) -> float:
    """Задержка передачи данных: объём (бит) / скорость канала. (3.1)"""
    return D * 8.0 / R


def t_computation(C: float, F: float) -> float:
    """Задержка вычисления: сложность / производительность узла. (3.2)"""
    return C / F


def t_queue(rho: float, mean_service: float) -> float:
    """
    Ожидание в очереди по M/M/1: Wq = ρ/(1-ρ)·E[обслуживание]. (3.3)
    rho clamp-ится < 1, иначе очередь бесконечна.
    """
    rho = min(max(rho, 0.0), 0.999)
    if rho <= 0.0:
        return 0.0
    return rho / (1.0 - rho) * mean_service


def t_cold(is_cold: bool, T_cold: float) -> float:
    """Штраф холодного старта: T_cold если функция холодная, иначе 0. (НОВОЕ)"""
    return T_cold if is_cold else 0.0


def offload_delay(task, F_node: float, rho: float, mean_service: float,
                  is_cold: bool, T_cold: float) -> float:
    """Суммарная задержка при выгрузке задачи на пограничный узел. (3.4)"""
    return (t_transmission(task.D, task.R)
            + t_computation(task.C, F_node)
            + t_queue(rho, mean_service)
            + t_cold(is_cold, T_cold))


def local_delay(task, F_loc: float) -> float:
    """Задержка локального исполнения на самом устройстве. (3.5)"""
    return t_computation(task.C, F_loc)
