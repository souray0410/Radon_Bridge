"""Ensure altered predictions and participant ordering cannot pass acceptance."""
import unittest
import numpy as np
from radonbridge.development_replay import compare_arrays

class ReplayChecks(unittest.TestCase):
    def setUp(self):
        self.expected = dict(ids=np.array([f'fake_{i}' for i in range(296)]), y=np.arange(296)%2,
                             cfp=np.tile([.7,.3],(296,1)), oct=np.tile([.4,.6],(296,1)))
    def test_identical(self):
        self.assertEqual(set(compare_arrays(self.expected,self.expected)), {'cfp','oct'})
    def test_reordered_ids(self):
        x={k:v.copy() for k,v in self.expected.items()};x['ids']=x['ids'][::-1]
        with self.assertRaisesRegex(ValueError,'ids'):compare_arrays(x,self.expected)
    def test_changed_labels(self):
        x={k:v.copy() for k,v in self.expected.items()};x['y'][0]=1
        with self.assertRaisesRegex(ValueError,'y order'):compare_arrays(x,self.expected)
    def test_probability_drift(self):
        x={k:v.copy() for k,v in self.expected.items()};x['cfp'][0]=[.69,.31]
        with self.assertRaisesRegex(ValueError,'probability replay'):compare_arrays(x,self.expected)
    def test_tiny_change_flips_argmax(self):
        x={k:v.copy() for k,v in self.expected.items()}
        x['cfp'][0]=[.5-1e-8,.5+1e-8];self.expected['cfp'][0]=[.5+1e-8,.5-1e-8]
        with self.assertRaisesRegex(ValueError,'decisions'):compare_arrays(x,self.expected)

if __name__=='__main__':unittest.main()
