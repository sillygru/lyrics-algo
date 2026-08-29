"""
Neural Lyrics Engine: Multi-layer residual network for word duration weighting and pause prediction.
"""

import math
import random

class NeuralLyricsEngine:
    """30-Dimensional Residual Multi-Layer Neural Network with Skip Connections."""

    def __init__(self, in_dim=30, h1=36, h2=20):
        self.in_dim = in_dim
        self.h1 = h1
        self.h2 = h2

        scale1 = math.sqrt(2.0 / in_dim)
        self.W1 = [[random.gauss(0, scale1) for _ in range(h1)] for _ in range(in_dim)]
        self.b1 = [0.01 for _ in range(h1)]

        scale2 = math.sqrt(2.0 / h1)
        self.W2 = [[random.gauss(0, scale2) for _ in range(h2)] for _ in range(h1)]
        self.b2 = [0.01 for _ in range(h2)]

        scale3 = math.sqrt(2.0 / h2)
        self.W3 = [random.gauss(0, scale3) for _ in range(h2)]
        self.b3 = 0.5

        # Linear direct skip from input to output
        self.W_skip = [random.gauss(0, 0.05) for _ in range(in_dim)]

        # Phrase Silence Net (5 inputs -> 8 hidden -> 1 output)
        self.W_pause = [[random.gauss(0, 0.4) for _ in range(8)] for _ in range(5)]
        self.b_pause = [0.01 for _ in range(8)]
        self.W_pause_out = [random.gauss(0, 0.4) for _ in range(8)]
        self.b_pause_out = -0.8

        self.comma_scale = 0.30
        self.stop_scale = 0.754
        self.onset_snap_strength = 0.40
        self.vad_trim_threshold = 0.08
        self.dtw_tightness = 1.5
        self.rms_weight = 0.40

    def forward_word(self, x):
        """Deterministic feedforward pass mapping 30-dim word feature vector to continuous duration weight."""
        h1 = [0.0] * self.h1
        for j in range(self.h1):
            s = self.b1[j]
            for i in range(min(len(x), self.in_dim)):
                s += x[i] * self.W1[i][j]
            h1[j] = s if s > 0 else 0.1 * s

        h2 = [0.0] * self.h2
        for j in range(self.h2):
            s = self.b2[j]
            for i in range(self.h1):
                s += h1[i] * self.W2[i][j]
            h2[j] = s if s > 0 else 0.1 * s

        out = self.b3
        for i in range(self.h2):
            out += h2[i] * self.W3[i]

        for i in range(min(len(x), self.in_dim)):
            out += x[i] * self.W_skip[i]

        if out > 20:
            return out
        elif out < -20:
            return 1e-4
        return math.log1p(math.exp(out))

    def forward_pause_ratio(self, p_feat):
        """Deterministic pass predicting phrase-level trailing silence ratio."""
        h = [0.0] * 8
        for j in range(8):
            s = self.b_pause[j]
            for i in range(min(len(p_feat), 5)):
                s += p_feat[i] * self.W_pause[i][j]
            h[j] = s if s > 0 else 0.1 * s
        out = self.b_pause_out
        for i in range(8):
            out += h[i] * self.W_pause_out[i]
        sig = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, out))))
        return sig * 0.50

    def get_params_dict(self):
        return {
            'in_dim': self.in_dim, 'h1': self.h1, 'h2': self.h2,
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'W_skip': self.W_skip,
            'W_pause': self.W_pause, 'b_pause': self.b_pause,
            'W_pause_out': self.W_pause_out, 'b_pause_out': self.b_pause_out,
            'comma_scale': self.comma_scale,
            'stop_scale': self.stop_scale,
            'onset_snap_strength': self.onset_snap_strength,
            'vad_trim_threshold': self.vad_trim_threshold,
            'dtw_tightness': self.dtw_tightness,
            'rms_weight': self.rms_weight,
        }

    def set_params_dict(self, d):
        if 'in_dim' in d and d['in_dim'] == self.in_dim and d['h1'] == self.h1 and d['h2'] == self.h2:
            self.W1 = [list(r) for r in d['W1']]
            self.b1 = list(d['b1'])
            self.W2 = [list(r) for r in d['W2']]
            self.b2 = list(d['b2'])
            self.W3 = list(d['W3'])
            self.b3 = d['b3']
            self.W_skip = list(d.get('W_skip', [0.0]*self.in_dim))
            if 'W_pause' in d and len(d['W_pause']) == 5:
                self.W_pause = [list(r) for r in d['W_pause']]
                self.b_pause = list(d['b_pause'])
                self.W_pause_out = list(d['W_pause_out'])
                self.b_pause_out = d['b_pause_out']
            self.comma_scale = d.get('comma_scale', 0.30)
            self.stop_scale = d.get('stop_scale', 0.754)
            self.onset_snap_strength = d.get('onset_snap_strength', 0.40)
            self.vad_trim_threshold = d.get('vad_trim_threshold', 0.08)
            self.dtw_tightness = d.get('dtw_tightness', 1.5)
            self.rms_weight = d.get('rms_weight', 0.40)

    def get_flat_vector(self):
        vec = []
        for row in self.W1: vec.extend(row)
        vec.extend(self.b1)
        for row in self.W2: vec.extend(row)
        vec.extend(self.b2)
        vec.extend(self.W3)
        vec.append(self.b3)
        vec.extend(self.W_skip)
        for row in self.W_pause: vec.extend(row)
        vec.extend(self.b_pause)
        vec.extend(self.W_pause_out)
        vec.append(self.b_pause_out)
        vec.extend([
            self.comma_scale,
            self.stop_scale,
            self.onset_snap_strength,
            self.vad_trim_threshold,
            self.dtw_tightness,
            self.rms_weight,
        ])
        return vec

    def set_flat_vector(self, vec):
        idx = 0
        for i in range(self.in_dim):
            for j in range(self.h1):
                self.W1[i][j] = vec[idx]; idx += 1
        for j in range(self.h1):
            self.b1[j] = vec[idx]; idx += 1
        for i in range(self.h1):
            for j in range(self.h2):
                self.W2[i][j] = vec[idx]; idx += 1
        for j in range(self.h2):
            self.b2[j] = vec[idx]; idx += 1
        for i in range(self.h2):
            self.W3[i] = vec[idx]; idx += 1
        self.b3 = vec[idx]; idx += 1
        for i in range(self.in_dim):
            self.W_skip[i] = vec[idx]; idx += 1
        for i in range(5):
            for j in range(8):
                self.W_pause[i][j] = vec[idx]; idx += 1
        for j in range(8):
            self.b_pause[j] = vec[idx]; idx += 1
        for i in range(8):
            self.W_pause_out[i] = vec[idx]; idx += 1
        self.b_pause_out = vec[idx]; idx += 1
        self.comma_scale = max(0.10, min(3.0, vec[idx])); idx += 1
        self.stop_scale = max(0.20, min(4.0, vec[idx])); idx += 1
        self.onset_snap_strength = max(0.05, min(0.95, vec[idx])); idx += 1
        self.vad_trim_threshold = max(0.02, min(0.30, vec[idx])); idx += 1
        self.dtw_tightness = max(0.50, min(4.0, vec[idx])); idx += 1
        self.rms_weight = max(0.05, min(1.50, vec[idx])); idx += 1

    def get_layer_scales(self):
        scales = []
        scales.extend([1.0 / math.sqrt(self.in_dim)] * (self.in_dim * self.h1))
        scales.extend([0.50] * self.h1)
        scales.extend([1.0 / math.sqrt(self.h1)] * (self.h1 * self.h2))
        scales.extend([0.50] * self.h2)
        scales.extend([1.0 / math.sqrt(self.h2)] * self.h2)
        scales.append(0.50)
        scales.extend([0.25] * self.in_dim)
        scales.extend([0.45] * (5 * 8))
        scales.extend([0.50] * 8)
        scales.extend([0.45] * 8)
        scales.append(0.50)
        scales.extend([1.80, 1.80, 1.50, 1.20, 1.20, 1.20])
        return scales

    def load_checkpoint(self, path: str):
        """Loads learned model parameters from a JSON checkpoint file."""
        import os
        import json
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            self.set_params_dict(data.get('neural_parameters', data))
            return True
        return False

    @classmethod
    def load_default(cls):
        """Creates a NeuralLyricsEngine instance loaded with packaged default weights."""
        import os
        engine = cls()
        package_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(package_dir, 'learned_parameters.json'),
            os.path.join(os.path.dirname(package_dir), 'learned_parameters.json'),
        ]
        for p in candidates:
            if engine.load_checkpoint(p):
                break
        return engine

