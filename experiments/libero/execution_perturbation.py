"""A bounded actuator disturbance, ending before the residual is measured."""
import hashlib
import numpy as np


class ExecutionPerturbation:
    def __init__(self, kind='clean', strength=0., *, target_chunk=1, window=16,
                 direction=(1., 0., 0.)):
        if kind not in ('clean', 'delay', 'translation_bias'):
            raise ValueError(kind)
        if window < 1 or target_chunk < 0 or strength < 0:
            raise ValueError('Invalid disturbance extent')
        if kind == 'delay' and (int(strength) != strength or strength > window):
            raise ValueError('Delay must be an integer within the observation window')
        self.kind, self.strength = kind, float(strength)
        self.target_chunk, self.window = target_chunk, window
        self.direction = np.asarray(direction, dtype=np.float64)
        if self.direction.shape != (3,) or not np.isclose(np.linalg.norm(self.direction), 1.):
            raise ValueError('Translation direction must be a unit 3-vector')
        self.chunk, self.step, self.previous_gripper = -1, 0, -1.
        self.trace = []
        self.planned_sha256 = None

    def begin_chunk(self, actions):
        self.chunk += 1
        self.step = 0
        self.actions = np.asarray(actions, dtype=np.float64).copy()
        self.hold_gripper = self.previous_gripper
        if self.chunk == self.target_chunk:
            self.planned_sha256 = hashlib.sha256(self.actions.tobytes()).hexdigest()

    def apply(self, planned):
        planned = np.asarray(planned, dtype=np.float64)
        actual = planned.copy()
        if self.chunk == self.target_chunk and self.step < self.window:
            if self.kind == 'delay':
                delay = int(self.strength)
                if self.step < delay:
                    actual[:6] = 0.
                    actual[-1] = self.hold_gripper
                else:
                    actual = self.actions[self.step-delay].copy()
            elif self.kind == 'translation_bias':
                actual[:3] = np.clip(actual[:3] + self.strength*self.direction, -1., 1.)
            self.trace.append(dict(step=self.step, planned=planned.tolist(), executed=actual.tolist()))
        # At step 16 the original command at index 16 resumes. Nothing later is perturbed.
        self.previous_gripper = float(actual[-1])
        self.step += 1
        return actual.tolist()

    def report(self):
        return dict(kind=self.kind, strength=self.strength, direction=self.direction.tolist(),
                    target_chunk=self.target_chunk, observation_window=self.window,
                    planned_action_sha256=self.planned_sha256,
                    altered_steps=sum(not np.array_equal(r['planned'], r['executed']) for r in self.trace),
                    action_trace=self.trace)


def select_calibrated_candidate(results):
    """Prefer mixed terminal outcomes; otherwise use the first failing level, or last level."""
    mixed = [r for r in results if 0 < r['failure_rate'] < 1]
    if mixed:
        return min(mixed, key=lambda r: (abs(r['failure_rate']-.5), r['order']))
    failing = [r for r in results if r['failure_rate'] > 0]
    return failing[0] if failing else results[-1]
