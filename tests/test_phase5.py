"""Final report provenance, references and truthful deliverable contracts."""
import csv
import hashlib
import json
from pathlib import Path
import re
import unittest
from datasets import OFFICIAL_DATASETS

ROOT=Path(__file__).resolve().parents[1]
class Phase5Tests(unittest.TestCase):
    def test_all_official_report_figures_exist_and_are_unique(self):
        with (ROOT/'reports/figure_index.csv').open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),13*len(OFFICIAL_DATASETS))
        self.assertEqual(len({r['path'] for r in rows}),13*len(OFFICIAL_DATASETS))
        report=(ROOT/'reports/CRISP_DM_REPORT.md').read_text(encoding='utf-8')
        for i,row in enumerate(rows,1):
            self.assertEqual(int(row['figure_number']),i)
            self.assertTrue((ROOT/'reports'/row['path']).is_file())
            self.assertIn(']('+row['path']+')',report)
            self.assertNotIn('???',row['description'])
        self.assertIn('相關矩陣',rows[10]['description'])
        self.assertIn('自相關',rows[11]['description'])
        self.assertIn('預測區間',rows[12]['description'])
    def test_report_values_and_hashes_come_from_current_runs(self):
        report=(ROOT/'reports/CRISP_DM_REPORT.md').read_text(encoding='utf-8')
        manifest=json.loads((ROOT/'reports/phase5/report_manifest.json').read_text(encoding='utf-8'))
        for key in OFFICIAL_DATASETS:
            path=ROOT/'results'/key/'FAST/run.json'
            run=json.loads(path.read_text(encoding='utf-8'))
            entry=next(r for r in manifest['datasets'] if r['id']==key)
            self.assertEqual(entry['run_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            for m in run['methods']:
                self.assertIn(format(m['test_metrics']['rmse'],'.4f'),report)
                self.assertIn(format(m['test_metrics']['r2'],'.4f'),report)
            self.assertIn(run['source_sha256'],report)
    def test_local_markdown_links_resolve(self):
        for path in [ROOT/'README.md',ROOT/'PHASE5.md',ROOT/'reports/CRISP_DM_REPORT.md',
                     ROOT/'reports/FIGURE_INDEX.md',ROOT/'reports/AI_ASSISTANCE.md']:
            text=path.read_text(encoding='utf-8')
            self.assertNotIn('\ufffd',text)
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if target.startswith(('http:','https:','#')):continue
                self.assertTrue((path.parent/target.split('#')[0]).exists(),(path,target))
    def test_ai_records_preserve_supplied_summary_and_pending_citations(self):
        text=(ROOT/'reports/AI_ASSISTANCE.md').read_text(encoding='utf-8')
        self.assertIn('100',text)
        self.assertIn('使用者提供',text)
        self.assertIn('來源引用',text)
        self.assertIn('RFE+VIF',text)
        supplied=(ROOT/'docs/NOTEBOOKLM_RESEARCH.md').read_text(encoding='utf-8')
        paragraph=next(line[2:] for line in supplied.splitlines() if line.startswith('> '))
        self.assertGreaterEqual(len(re.findall(r'[\u4e00-\u9fff]',paragraph)),100)
        self.assertIn(paragraph,(ROOT/'README.md').read_text(encoding='utf-8'))

        self.assertIn('待填',text)
        self.assertIn('真實 ChatGPT',text)
        self.assertIn('PDF',text)
    def test_readme_is_generated_from_template_with_real_tables(self):
        text=(ROOT/'README.md').read_text(encoding='utf-8')
        self.assertNotIn('{{',text)
        for key in ['Student Performance','Appliances Energy','Correlation','RFECV','LASSO','FULL','FAST']:
            self.assertIn(key,text)
        self.assertIn('不是', (ROOT/'reports/CRISP_DM_REPORT.md').read_text(encoding='utf-8'))
        self.assertIn('不能直接',text)
    def test_dataset_provenance_matches_preserved_authentic_files(self):
        student=ROOT/'datasets/student_performance'
        source=json.loads((student/'student-mat.source.json').read_text(encoding='utf-8'))
        self.assertEqual(source['sha256'],hashlib.sha256((student/'student-mat.csv').read_bytes()).hexdigest())
        energy=ROOT/'datasets/appliances_energy'
        source=json.loads((energy/'source.json').read_text(encoding='utf-8'))
        self.assertEqual(source['files'][0]['sha256'],hashlib.sha256((energy/'energydata_complete.csv').read_bytes()).hexdigest())

    def test_phase2_and_phase3_use_same_effective_rank(self):
        import joblib
        import numpy as np
        from datasets import load_dataset
        from regression_pipeline import fitted_design_rank, regression_metrics
        from regression_diagnostics import selected_matrix, design_information, rank_metrics
        for key,config in OFFICIAL_DATASETS.items():
            run=json.loads((ROOT/'results'/key/'FAST/run.json').read_text(encoding='utf-8'))
            data=load_dataset(config)
            train=run['train_row_indices'];test=run['test_row_indices']
            for m in run['methods']:
                model=joblib.load(ROOT/'results'/key/'FAST/models'/(m['method'].lower().replace(' ','_')+'.joblib'))
                rank=fitted_design_rank(model,data.X.iloc[train])
                self.assertEqual(rank,design_information(selected_matrix(model,data.X.iloc[train]))['rank'])
                expected=rank_metrics(data.y.iloc[test],model.predict(data.X.iloc[test]),rank)
                actual=regression_metrics(data.y.iloc[test],model.predict(data.X.iloc[test]),m['selected_encoded_count'],fitted_rank=rank)
                if expected['adjusted_r2'] is None:
                    self.assertIsNone(actual['adjusted_r2'])
                    self.assertIsNone(m['test_metrics']['adjusted_r2'])
                else:
                    self.assertAlmostEqual(actual['adjusted_r2'],expected['adjusted_r2'])
                    self.assertAlmostEqual(m['test_metrics']['adjusted_r2'],expected['adjusted_r2'])
    def test_effective_rank_differs_from_aliased_column_count(self):
        from regression_pipeline import regression_metrics
        nominal=regression_metrics([1,2,3,4],[1,2,3,3],3)
        effective=regression_metrics([1,2,3,4],[1,2,3,3],3,fitted_rank=2)
        self.assertIsNone(nominal['adjusted_r2'])
        self.assertAlmostEqual(effective['adjusted_r2'],0.7)

if __name__=='__main__':
    unittest.main()
