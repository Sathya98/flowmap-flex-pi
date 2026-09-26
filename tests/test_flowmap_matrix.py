"""Checkpoint lineage and experiment counting; no ML dependencies."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'flowmap_matrix', Path(__file__).resolve().parents[1] / 'scripts/flowmap_matrix.py')
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class MatrixTests(unittest.TestCase):
    def test_main_lineage_and_full_joint(self):
        plan = matrix.build_plan(matrix.parse_args(['--agibot-checkpoint', '/models/agibot.pt']))
        training = [r for r in plan if r['train_command']]
        self.assertEqual(len(plan), 16)
        self.assertEqual(len(training), 14)
        for row in training:
            self.assertEqual(set(row['streams']), {'action', 'video', 'dino', 'pointmap'})
            self.assertEqual(row['mode'], 'full')
            if row['method'] == 'self_distillation':
                self.assertEqual(row['initial_checkpoint'], '/models/agibot.pt')
                self.assertIsNone(row['teacher_checkpoint'])
                self.assertEqual(row['evaluation_ema_decay'], .9999)
                self.assertIn('ema_0.9999', row['evaluation_checkpoint_pattern'])
                self.assertIn('model.flow_map.self_diagonal_fraction=0.75', row['train_command'])
            else:
                self.assertEqual(row['initial_checkpoint'], row['teacher_checkpoint'])
                self.assertIn('model.flow_map.distill_ema=true', row['train_command'])
                self.assertEqual(row['loss_weighting'], 'fixed')
                if row['objective'] == 'lmd':
                    self.assertEqual(row['lmd_teacher_gradient'], 'full')
                if row['objective'] == 'pfmm':
                    self.assertEqual(row['pfmm_loss_space'], 'endpoint')
        libero = [r for r in plan if r['benchmark_train'] == 'libero']
        self.assertTrue(all(r['evaluations'] == ['libero', 'libero_plus'] for r in libero))

    def test_no_silent_initialization_fallback(self):
        plan = matrix.build_plan(matrix.parse_args([]))
        rows = [r for r in plan if r['method'] == 'self_distillation']
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r['train_command'] is None and r['status'] == 'needs_agibot_checkpoint' for r in rows))
        teacher = plan[0]['checkpoint']
        with self.assertRaisesRegex(ValueError, 'task-finetuned'):
            matrix.build_plan(matrix.parse_args(['--agibot-checkpoint', teacher]))

    def test_tables_have_no_invented_results(self):
        plan = matrix.build_plan(matrix.parse_args([]))
        with tempfile.TemporaryDirectory() as folder:
            matrix.write_tables(plan, folder)
            for benchmark in matrix.METRICS:
                with (Path(folder) / (benchmark + '.csv')).open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 40)
                self.assertTrue(all(r['status'] == 'not_run' for r in rows))
                self.assertTrue(all(not r[m] for r in rows for m in matrix.METRICS[benchmark]))
            with self.assertRaises(FileExistsError):
                matrix.write_tables(plan, folder)


if __name__ == '__main__':
    unittest.main()
