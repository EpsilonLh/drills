"""Persist validation output without rewriting any historical experiment file."""
import contextlib
import io
import sys
import time
import unittest

from study import ROOT,HERE,sources,dump,now,sha


def main():
    started=now();timer=time.perf_counter();stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (HERE/'tests.log').write_text(stream.getvalue())
    print(stream.getvalue(),end='')
    dump(HERE/'test-results.json',dict(command=[sys.executable,'-B',str(HERE/'check.py')],
        started_at=started,finished_at=now(),elapsed_seconds=time.perf_counter()-timer,
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),successful=result.wasSuccessful(),source_hashes=sources(),
        test_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tests').glob('test_*.py')},
        log_sha256=sha(HERE/'tests.log')))
    if not result.wasSuccessful() or result.skipped:raise SystemExit(1)


if __name__=='__main__':main()
