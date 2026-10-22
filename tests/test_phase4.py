"""Phase 4 artifact provenance and static deployment contract."""
import hashlib
import json
import re
import unittest
from pathlib import Path
from datasets import OFFICIAL_DATASETS
ROOT=Path(__file__).resolve().parents[1]
DIST=ROOT/'dashboard/dist'
class Phase4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=json.loads((DIST/'data.json').read_text(encoding='utf-8'))
    def test_official_registry_and_feature_metadata(self):
        self.assertEqual(set(OFFICIAL_DATASETS),{'student_performance','appliances_energy'})
        self.assertEqual(set(self.bundle['datasets']),set(OFFICIAL_DATASETS))
        for key,config in OFFICIAL_DATASETS.items():
            d=self.bundle['datasets'][key]
            self.assertEqual(d['metadata']['originalFeatureCount'],config.original_feature_count)
            self.assertEqual({f['name'] for f in d['inputs']},set(config.feature_names))
            self.assertNotIn(config.target,config.feature_names)
            self.assertEqual(d['metadata']['unit'],config.target_unit)
            self.assertEqual(len(d['preprocessing']['featureNames']),d['metadata']['encodedFeatureCount'])
    def test_actual_run_provenance_and_training_cv_recommendation(self):
        for key,config in OFFICIAL_DATASETS.items():
            d=self.bundle['datasets'][key];path=ROOT/'results'/key/d['mode']/'run.json'
            run=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(d['sourceRunSha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(run['source_sha256'],hashlib.sha256(config.path.read_bytes()).hexdigest())
            self.assertEqual(d['defaultMethod'],run['best_model']['method'])
            self.assertEqual(len(d['comparison']),6)
            for row in run['methods']:
                model=next(m for m in d['models'] if m['method']==row['method'] and m['isOfficial'])
                self.assertAlmostEqual(model['cvRmse'],row['best_cv_rmse'])
                self.assertAlmostEqual(model['cvStd'],row['cv_std'])
                self.assertEqual(model['params'],row['best_params'])
                self.assertEqual(model['features'],row['selected_encoded_names'])
                self.assertEqual(model['k'],len(model['weights']))
    def test_json_and_direct_file_bundle_are_identical(self):
        script=(DIST/'data.js').read_text(encoding='utf-8').strip()
        self.assertTrue(script.startswith('window.DASHBOARD_DATA = '))
        self.assertEqual(json.loads(script.split(' = ',1)[1][:-1]),self.bundle)
    def test_relative_asset_versions_match_contents(self):
        html=(DIST/'index.html').read_text(encoding='utf-8')
        for name in ['app.js','data.js','prediction.js','styles.css']:
            digest=hashlib.sha256((DIST/name).read_bytes()).hexdigest()[:12]
            self.assertIn(name+'?v='+digest,html)
            self.assertNotRegex(html,re.escape('/'+name))
if __name__=='__main__':unittest.main()
