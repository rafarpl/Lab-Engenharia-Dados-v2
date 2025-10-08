import os
import sys
import py_compile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print('Repository root:', ROOT)

errors = []

def check_file(path):
    if not os.path.exists(path):
        errors.append(f'MISSING: {path}')
        print(errors[-1])
    else:
        print('OK:', path)

def compile_py(path):
    try:
        py_compile.compile(path, doraise=True)
        print('COMPILE OK:', path)
    except py_compile.PyCompileError as e:
        errors.append(f'COMPILE ERROR: {path} -> {e}')
        print(errors[-1])

files_to_check = [
    os.path.join(ROOT, '.gitpod.yml'),
    os.path.join(ROOT, '.gitpod', 'start_smoke_test.sh'),
    os.path.join(ROOT, 'airflow', 'dags', 'dag_orchestrator_pipeline.py'),
    os.path.join(ROOT, 'airflow', 'utils', 'pipeline_utils.py'),
    os.path.join(ROOT, 'airflow', 'dags', 'dag_bronze_to_silver.py'),
    os.path.join(ROOT, 'airflow', 'dags', 'dag_trusted_to_refined.py'),
]

for f in files_to_check:
    check_file(f)

# Try to compile Python files (only those present)
for f in files_to_check:
    if os.path.exists(f) and f.endswith('.py'):
        compile_py(f)

print('\nSummary:')
if errors:
    print('SMOKE FAILED with errors:')
    for e in errors:
        print(' -', e)
    sys.exit(1)
else:
    print('SMOKE PASSED: basic file checks and syntax OK')
    sys.exit(0)
    
