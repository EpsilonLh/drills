"""Run the actual unittest suite and persist its command, source hashes and output."""
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

from common import ROOT, HERE, dump, now, sha, sources


def main():
    stream = io.StringIO()
    started = now()
    timer = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    (HERE / 'tests.log').write_text(output)
    print(output, end='')
    dump(HERE / 'test-results.json', dict(command=[sys.executable, '-B', str(Path(__file__).resolve())],
        started_at=started, finished_at=now(), elapsed_seconds=time.perf_counter() - timer,
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), successful=result.wasSuccessful(),
        source_hashes={**sources(), **{str(p.relative_to(ROOT)): sha(p) for p in HERE.glob('*.py')}},
        test_hashes={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'tests').glob('test_*.py')},
        log_sha256=sha(HERE / 'tests.log')))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
