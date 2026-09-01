"""Tests for the shared pipeline entry point and the Earth Engine project lookup.

``run_pipeline.run_all`` is what both the CLI and the Kaggle notebook call, so the
things worth pinning are the ones that differ between those two callers: which
directory the stages run in, whether the caller's directory survives, and that the
stage scripts are addressed absolutely rather than relative to the cwd.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import preflight
import run_pipeline


class WorkingDirectoryTests(unittest.TestCase):
    def test_stages_run_in_the_repo_root_not_the_scripts_dir(self):
        # The per-species store (data/) and the raster caches (soil/, treecover/,
        # dem/ ...) are all relative to the repo root. Running from scripts/ would
        # resolve DATA_DIR='data' to scripts/data and silently miss the real store.
        self.assertEqual(run_pipeline.ROOT_DIR, run_pipeline.SCRIPTS_DIR.parent)
        self.assertTrue((run_pipeline.ROOT_DIR / 'scripts').is_dir())

    def test_working_directory_restores_the_previous_cwd(self):
        before = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            with run_pipeline.working_directory(tmp) as entered:
                self.assertEqual(Path(os.getcwd()).resolve(), Path(tmp).resolve())
                self.assertEqual(Path(entered).resolve(), Path(tmp).resolve())
        self.assertEqual(os.getcwd(), before)

    def test_working_directory_restores_even_when_a_stage_raises(self):
        before = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                with run_pipeline.working_directory(tmp):
                    raise SystemExit('stage failed')
        self.assertEqual(os.getcwd(), before)

    def test_run_all_runs_from_the_repo_root_by_default(self):
        seen = {}

        def fake_stages(python_executable=None):
            seen['cwd'] = os.getcwd()

        with mock.patch.object(run_pipeline, '_run_stages', fake_stages):
            run_pipeline.run_all()

        self.assertEqual(Path(seen['cwd']).resolve(), run_pipeline.ROOT_DIR.resolve())

    def test_run_all_honours_an_explicit_root(self):
        seen = {}

        def fake_stages(python_executable=None):
            seen['cwd'] = os.getcwd()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run_pipeline, '_run_stages', fake_stages):
                run_pipeline.run_all(root=tmp)
            self.assertEqual(Path(seen['cwd']).resolve(), Path(tmp).resolve())


class RunStepTests(unittest.TestCase):
    def test_stage_scripts_are_addressed_absolutely(self):
        # run_all() runs from the repo root, so a bare "iNat.py" would not resolve.
        with mock.patch.object(subprocess, 'run',
                               return_value=mock.Mock(returncode=0)) as runner:
            run_pipeline.run_step('Fetch', '/usr/bin/python3', 'iNat.py')

        cmd = runner.call_args.args[0]
        self.assertEqual(cmd[0], '/usr/bin/python3')
        self.assertEqual(Path(cmd[1]), run_pipeline.SCRIPTS_DIR / 'iNat.py')
        self.assertTrue(Path(cmd[1]).is_absolute())

    def test_a_failing_stage_stops_the_pipeline(self):
        with mock.patch.object(subprocess, 'run', return_value=mock.Mock(returncode=2)):
            with self.assertRaises(SystemExit):
                run_pipeline.run_step('Fetch', '/usr/bin/python3', 'iNat.py')

    def test_main_delegates_to_run_all(self):
        # The CLI must not carry its own copy of the sequence.
        with mock.patch.object(run_pipeline, 'run_all') as runner:
            run_pipeline.main()
        runner.assert_called_once_with()


class EarthEngineProjectTests(unittest.TestCase):
    def test_environment_variable_wins(self):
        with mock.patch.dict(os.environ, {'EARTHENGINE_PROJECT': 'my-proj'}, clear=True):
            project, source = preflight.resolve_earthengine_project()
        self.assertEqual(project, 'my-proj')
        self.assertEqual(source, 'EARTHENGINE_PROJECT')

    def test_falls_back_to_the_stored_credential(self):
        with tempfile.TemporaryDirectory() as home:
            cred = Path(home) / '.config' / 'earthengine' / 'credentials'
            cred.parent.mkdir(parents=True)
            cred.write_text('{"project": "cred-proj"}', encoding='utf-8')
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(Path, 'home', return_value=Path(home)):
                project, source = preflight.resolve_earthengine_project()
        self.assertEqual(project, 'cred-proj')
        self.assertIn('credentials', source)

    def test_falls_back_to_gcloud(self):
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(Path, 'home', return_value=Path(home)), \
                 mock.patch.object(subprocess, 'run',
                                   return_value=mock.Mock(stdout='gcloud-proj\n')):
                project, source = preflight.resolve_earthengine_project()
        self.assertEqual(project, 'gcloud-proj')
        self.assertIn('gcloud', source)

    def test_unset_gcloud_project_is_not_treated_as_a_value(self):
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(Path, 'home', return_value=Path(home)), \
                 mock.patch.object(subprocess, 'run',
                                   return_value=mock.Mock(stdout='(unset)\n')):
                project, source = preflight.resolve_earthengine_project()
        self.assertIsNone(project)
        self.assertIsNone(source)

    def test_a_corrupt_credential_file_does_not_raise(self):
        with tempfile.TemporaryDirectory() as home:
            cred = Path(home) / '.config' / 'earthengine' / 'credentials'
            cred.parent.mkdir(parents=True)
            cred.write_text('not json at all', encoding='utf-8')
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(Path, 'home', return_value=Path(home)), \
                 mock.patch.object(subprocess, 'run', side_effect=OSError('no gcloud')):
                project, _source = preflight.resolve_earthengine_project()
        self.assertIsNone(project)

    def test_reporter_explains_where_to_look_when_unconfigured(self):
        with mock.patch.object(preflight, 'resolve_earthengine_project',
                               return_value=(None, None)), \
             mock.patch('builtins.print') as printer:
            result = preflight.print_earthengine_project()

        self.assertIsNone(result)
        printed = ' '.join(str(c.args[0]) for c in printer.call_args_list if c.args)
        self.assertIn('console.cloud.google.com', printed)
        self.assertIn('code.earthengine.google.com', printed)


if __name__ == '__main__':
    unittest.main()
