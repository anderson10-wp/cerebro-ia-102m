import copy,tempfile,unittest,urllib.error
from pathlib import Path
import experiment_analyzer as a
class Tests(unittest.TestCase):
    def test_valid(self): a.validate(a.example_plan(),{'facts':{'general_dev':{},'technical_dev':{}}})
    def test_reject(self):
        for field,value in [('seconds',601),('seconds',True),('batch_size',8),('learning_rate',float('nan')),('device','cuda')]:
            plan=a.example_plan(); plan['experiment'][field]=value
            with self.assertRaises(ValueError): a.validate(plan,{'facts':{'general_dev':{},'technical_dev':{}}})
        for key,value in [('execute_training',True),('action','run_shell'),('evidence_ids',['invented']),('unexpected','command')]:
            plan=a.example_plan(); plan[key]=value
            with self.assertRaises(ValueError): a.validate(plan,{'facts':{'general_dev':{},'technical_dev':{}}})
    def test_outputs_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            result=a.execute(folder,lambda facts:(a.example_plan(),{},'offline-fixture'))
            self.assertEqual(result['status'],'proposal_validated'); self.assertFalse(result['training_started'])
        with tempfile.TemporaryDirectory() as tmp:
            def fail(facts): raise urllib.error.HTTPError('https://api.groq.com',429,'limited',None,None)
            result=a.execute(Path(tmp),fail)
            self.assertEqual(result['status'],'api_error'); self.assertFalse(result['training_started'])
        with tempfile.TemporaryDirectory() as tmp:
            result=a.execute(Path(tmp),lambda facts:({'command':'invalid'},{},'offline-fixture'))
            self.assertEqual(result['status'],'rejected'); self.assertFalse((Path(tmp)/'PLANO.txt').exists())
if __name__=='__main__': unittest.main()
