"""Persist new 50-step checks and regressions of the unchanged original suite."""
import io
from pathlib import Path
import subprocess
import sys
import time
import unittest
from common import ROOT,HERE,dump,now,sha,sources


def main():
    started=now(); timer=time.perf_counter(); stream=io.StringIO()
    for path in HERE.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    tested_hashes={**sources(), **{str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}}
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_suite.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    old_command=[sys.executable,'-B','-m','unittest','discover','-s','tests','-v']
    old=subprocess.run(old_command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output=stream.getvalue()+'\nUnchanged original regression suite:\n'+old.stdout
    (HERE/'tests.log').write_text(output); print(output,flush=True)
    unchanged=all(sha(ROOT/name)==digest for name,digest in tested_hashes.items())
    dump(HERE/'test-results.json',dict(command=[sys.executable,'-B',str(Path(__file__).resolve())],
        started_at=started,finished_at=now(),elapsed_seconds=time.perf_counter()-timer,
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        original_regression_command=old_command,original_regression_exit_code=old.returncode,
        successful=result.wasSuccessful() and old.returncode==0 and unchanged,
        sources_unchanged_during_checks=unchanged,source_hashes=tested_hashes,
        log_sha256=sha(HERE/'tests.log')))
    if not result.wasSuccessful() or old.returncode or not unchanged:
        raise SystemExit(1)


if __name__=='__main__':
    main()
