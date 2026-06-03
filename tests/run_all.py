"""
Запуск всех юнит-тестов БЕЗ pytest.

Находит все функции test_* во всех модулях test_*.py рядом, выполняет их,
печатает PASS/FAIL. Возвращает код выхода 1 при любом провале.

    python tests/run_all.py
"""
import os, sys, importlib, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # корень проекта
sys.path.insert(0, HERE)                     # сами тесты


def main():
    test_files = [f[:-3] for f in os.listdir(HERE)
                  if f.startswith('test_') and f.endswith('.py')]
    total = passed = 0
    failures = []
    for mod_name in sorted(test_files):
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith('test_'):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            total += 1
            try:
                fn()
                passed += 1
                print(f"  PASS  {mod_name}.{name}")
            except Exception as e:
                failures.append((mod_name, name, e))
                print(f"  FAIL  {mod_name}.{name}: {e}")
                traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"  {passed}/{total} тестов прошло")
    print("=" * 50)
    if failures:
        print("ПРОВАЛЫ:")
        for m, n, e in failures:
            print(f"  - {m}.{n}: {e}")
        return 1
    print("Все тесты зелёные.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
