"""Build report-ready Traditional Chinese Markdown from actual model artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datasets import OFFICIAL_DATASETS, load_dataset

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports'
REFS = """
1. [Student 原始資料／G1、G2 說明](https://archive.ics.uci.edu/dataset/320/student+performance)。
2. [指定 Student Kaggle](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set)：實際提供 Portuguese。
3. [Appliances Kaggle](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)。
4. [Appliances 原始 UCI](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)。
5. [scikit-learn Pipeline 與資料洩漏](https://scikit-learn.org/stable/common_pitfalls.html)。
6. [statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)。
7. [GitHub Pages 工作流程](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。
9. [Car Price Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)。
8. [專案原始碼](https://github.com/RayWu576/AIoT-HW2)。
"""
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def f(value, digits=4):
    return '不適用' if value is None else format(value, '.'+str(digits)+'f')
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])
def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip()+'\n', encoding='utf-8')

def prepare(mode='FAST'):
    overview, comparisons, timings, parts, figures, manifests = [], [], [], [], [], []
    for key, config in OFFICIAL_DATASETS.items():
        data=load_dataset(config)
        source=ROOT/'results'/key/mode
        run=read(source/'run.json')
        diag=read(REPORTS/'tables'/key/mode/'diagnostics.json')
        digest=hashlib.sha256((source/'run.json').read_bytes()).hexdigest()
        if digest!=diag['source_run_sha256']:
            raise ValueError('Stale diagnostics: '+key)
        if hashlib.sha256(config.path.read_bytes()).hexdigest()!=run['source_sha256']:
            raise ValueError('Changed source: '+key)
        student=key=='student_performance'
        title='學生表現' if student else '汽車價格' if key=='car_price' else '家電能源'
        best=next(m for m in run['methods'] if m['method']==run['best_model']['method'])
        evidence=diag['models'][best['method']]
        overview.append([title,len(data.y),len(data.raw_columns),config.original_feature_count,
                         best['encoded_input_count'],config.target,config.target_unit,
                         str(run['train_size'])+' / '+str(run['test_size'])])
        timings.append([title,f(run['runtime_seconds'],2),f(diag['runtime_seconds'],2),
                        f(run['runtime_seconds']+diag['runtime_seconds'],2),str(run['cache_namespace_preexisting'])])
        for m in run['methods']:
            comparisons.append([title,m['method'],m['selected_encoded_count'],f(m['best_cv_rmse']),
                                f(m['cv_std']),f(m['test_metrics']['rmse']),f(m['test_metrics']['r2'])])
        with (REPORTS/'tables'/key/mode/'evaluation_metrics.csv').open(encoding='utf-8',newline='') as stream:
            metrics=list(csv.DictReader(stream))
        metric_table=table(['方法','資料','encoded k','有效 rank','MAE','MSE','RMSE','R²','Adjusted R²'],
            [[r['method'],r['split'],r['selected_encoded_count'],r['fitted_design_rank'],
              *[f(float(r[n])) if r[n] else '不適用' for n in ['mae','mse','rmse','r2','adjusted_r2']]] for r in metrics])
        assumptions=[]
        interpretation={
            'linearity':'以殘差對 fitted 與分箱平均檢查線性。殘差均值跨度 >0.5 個殘差 SD 只是宣告的視覺篩檢，不是顯著性檢定；未觸發不能證明線性。',
            'independence':('學生來源行序不是時間，DW 只能描述該順序，不足以證實獨立；學校／群組相依仍可能存在。' if student else '能源殘差 DW 約 0.532、lag-1 ACF 約 0.734，顯示強烈正自相關；Ljung-Box 也支持序列相依，p=0 是浮點下溢，不是字面機率為零。'),
            'homoscedasticity':('Koenker-Breusch-Pagan 未拒絕等變異，但不等於證明等變異；選擇程序影響 nominal p-value。' if student else 'Koenker-Breusch-Pagan 提供反對等變異的證據；時間相依又限制傳統檢定 p-value 的解釋。'),
            'normality':'實際常態性檢定拒絕常態。學生與汽車採 Shapiro-Wilk，能源大樣本採 Jarque-Bera；大樣本可對小偏離很敏感，必須同看 Q-Q、直方圖、偏態及尾部。',
            'multicollinearity':'檢查含截距的 affine rank、VIF 與 condition。完整 one-hot 可造成 aliasing；個別不可識別係數不提供虛假的 SE/CI，不能只把原始特徵數當模型維度。'
        }
        if key=='car_price':
            interpretation['independence']='汽車原始行序不是時間；DW／ACF 只能作描述，不能證明觀測獨立。同廠牌或車型群組仍可能相依。'
            interpretation['homoscedasticity']='依下列真實 Koenker-Breusch-Pagan 統計判讀；選模及稀疏類別會限制 p 值解釋，未拒絕不等於證明等變異。'

        for a in evidence['assumptions']:
            assumptions.append('#### '+a['assumption']+' — '+a['method']+'\n\n'+interpretation[a['assumption']]+'\n\n實際統計：\n\n    '+
                json.dumps(a['statistic'],ensure_ascii=False)+'\n\n原始診斷解讀：'+a['interpretation']+'\n\n限制：'+a['limitations']+'\n')
        image_blocks=[]
        captions=['訓練／測試目標分布','訓練原始數值特徵相關矩陣','特徵數與 inner CV RMSE',
                  '特徵數與 inner validation R²','Baseline 與特徵選擇 RMSE 比較','測試實際值對預測值',
                  '訓練殘差對擬合值','訓練／測試殘差直方圖','訓練殘差常態 Q-Q 圖',
                  '選定模型係數','選定編碼特徵相關矩陣','訓練殘差自相關','測試平均信賴區間與個別預測區間']
        # The manifest order is authoritative; meaningful captions are keyed below.
        captions_by_name={
            'target_distribution_train_test':'訓練／測試目標分布',
            'feature_correlation_heatmap':'訓練特徵相關矩陣',
            'feature_count_vs_cv_rmse':'特徵數與 training CV RMSE',
            'feature_count_vs_validation_r2':'特徵數與 validation R²',
            'baseline_vs_feature_selection':'Baseline 與特徵選擇比較',
            'actual_vs_predicted':'測試實際值對預測值',
            'residuals_vs_fitted':'訓練殘差對擬合值',
            'residual_histogram_train_test':'訓練／測試殘差直方圖',
            'qq_plot':'訓練殘差 Q-Q 圖',
            'selected_feature_coefficients':'選定模型係數',
            'residual_autocorrelation':'依來源順序的訓練殘差自相關',
            'prediction_intervals':'測試平均信賴區間與個別預測區間',
            'vif_selected_features':'選定特徵 VIF',
        }
        for i,fig in enumerate(diag['figures']):
            path=ROOT/fig['path']
            if not path.is_file():raise FileNotFoundError(path)
            number=len(figures)+1
            desc=captions_by_name.get(fig['name'],captions[i])
            section='資料理解' if i<2 else '建模' if i<4 else '評估／回歸假設'
            rel=path.relative_to(REPORTS).as_posix()
            figures.append([number,key,path.name,desc,section,rel])
            image_blocks.append(f'![圖 {number}：{title} — {desc}]({rel})\n\n圖 {number}：{title}，{desc}；實際 {mode} 產物，{fig["dpi"]} dpi。')
        problem=('以背景、家庭、學習與出缺席資料預測數學 G3，供學習支持研究使用；不是因果評估或個人處分工具。G1/G2 是歷史學期成績，取得後是另一個較晚期情境，正式 early-30 沒有使用。'
                 if student else '由住宅同期感測及氣象估計 Appliances（Wh），探討節能估計的可行性。保留 lights；同期估計不代表可事先取得所有輸入的未來負載預報。')
        split=('seed=42，316/79 的 shuffled 80/20 holdout，inner/outer 為 shuffled KFold。'
               if config.split_strategy!='chronological' else '前 15,788 筆訓練，後 3,947 筆測試；inner/outer 均為 expanding TimeSeriesSplit，gap=0，不 shuffle。時間從 '+str(data.temporal_metadata.date.iloc[0])+' 至 '+str(data.temporal_metadata.date.iloc[-1])+'。')
        exclusions='G3/G1/G2' if student else 'Appliances、rv1、rv2；date 另存 metadata'
        outcome=('測試 R² 約 0.0795，顯示 early-30 預測訊號有限；Baseline 測試 RMSE 較低，仍不據此重選方法。'
                 if student else '測試 R² 為負，表示在此測試時段的描述性比較中，SSE 大於以測試目標平均值計算的 SST；不表示已建立可部署的平均值模型。Baseline 測試表現較佳仍不據此改選方法。自相關、右尾與異質變異使傳統 OLS 個別 PI 不宜當作已校準保證。')
        if key=='car_price':
            problem='以車輛規格與廠牌估計 price（USD），研究小樣本價格迴歸；這不是當前市場報價或因果效果。CarName 147 個原始名稱先作確定性廠牌擷取，22 個廠牌保留有意義資訊，避免車型名稱過度擬合。'
            split=f"seed={run['configuration']['seed']}，{run['train_size']}/{run['test_size']} shuffled 80/20 holdout；inner/outer 均為 shuffled KFold。"
            exclusions='car_ID 與 price；24 個原始候選輸入全部保留，CarName 在 Pipeline 中轉為廠牌'
            outcome=f"正式 training-CV 選定 {best['method']}（最終仍為 LinearRegression），Test RMSE={best['test_metrics']['rmse']:.4f} USD，R²={best['test_metrics']['r2']:.4f}。只有 205 筆觀測，類別稀疏、品牌群組與未知類別限制外推；Baseline 的 test 較佳也不依 test 改選方法。"

        parts.append(f"""## {title}：六階段 CRISP-DM

### 1. Business Understanding — 業務理解

{problem}

### 2. Data Understanding — 資料理解

實際 {len(data.y):,} 筆、{len(data.raw_columns)} 個 CSV 欄位、{config.original_feature_count} 個原始輸入，包含 {len(config.numeric_columns)} 數值／{len(config.categorical_columns)} 類別。
目標範圍 {data.y.min():.2f}–{data.y.max():.2f}，平均 {data.y.mean():.4f}，樣本標準差 {data.y.std():.4f}。
輸入缺失儲存格 {int(data.X.isna().sum().sum())} 個。沒有缺值也保留可重用的 fold-local imputer。

原始 CSV：{', '.join(data.raw_columns)}。

正式輸入：{', '.join(config.feature_names)}。

資料 SHA-256：{run['source_sha256']}。

### 3. Data Preparation — 資料準備

排除 {exclusions}。{split}
數值中位數補值與 StandardScaler；學生名目類別使用眾數補值與 OneHotEncoder(handle_unknown="ignore")。
全部處理、排名與選擇放在完整 Pipeline，由每個 training fold 擬合，不能先全資料 fit 再 CV。
來源索引保留，能源不排序、不回填未來觀測；非法 schema、數值、目標或時間戳明確拒絕。

### 4. Modeling — 建模

Baseline 與五種方法最終都使用 LinearRegression。{mode} 的 inner={run['configuration']['inner_folds']}、outer={run['configuration']['outer_folds']}。
內層選 k／alpha；外層評估完整訓練期選擇流程；全訓練部分再依 inner CV 固定正式方法，最後才評估 test。
推薦 **{best['method']}**，{best['selected_encoded_count']} 個 encoded 特徵，代表 {best['selected_original_count']} 個原始欄位；完整訓練編碼寬度 {best['encoded_input_count']}。

選中特徵：{', '.join(best['selected_encoded_names'])}。

Inner CV RMSE={best['best_cv_rmse']:.6f} ± {best['cv_std']:.6f}；
該方法 outer nested RMSE={best['nested_cv_rmse']:.6f} ± {best['nested_cv_std']:.6f}；
包含方法選擇的 workflow nested RMSE={run['workflow_nested_cv_rmse']:.6f} ± {run['workflow_nested_cv_std']:.6f}。
三者不是同一統計量；inner tuning score 不能冒充獨立泛化保證。

### 5. Evaluation — 評估

{metric_table}

Adjusted R² 使用 training affine design rank（含截距）；test 值只是描述性懲罰，不是重新擬合測試資料的推論。
自由度不足或常數目標時為空值。以下五項證據不代表因果，也不能自動宣稱假設全部成立。

{chr(10).join(assumptions)}

#### 信賴區間、預測區間與實際殘差分析

條件平均 mean CI 描述 E[Y|X]；個別觀測 PI 加入殘差變異，通常更寬。
使用同一已選定 sklearn 模型的設計與係數順序，透過 centered SVD 的等價滿秩基底與 statsmodels OLS 計算 t 區間。
學生與汽車另有 HC3、能源有 HAC（24 lag）漸近 mean CI；robust mean CI 不是 robust individual PI。
超出訓練 affine span 時不輸出可識別區間；不強行反轉奇異矩陣。
沒有校正特徵／方法選擇不確定性，沒有證實 nominal coverage；不任意裁切負 Wh 或超過 0–20 的區間。

實際區間 metadata：

    {json.dumps(evidence['intervals'],ensure_ascii=False)}

{outcome}

### 6. Deployment — 部署

既有靜態 DataLab 讀取 Python 真實產出的 schema-5 descriptor、係數、covariance、指標與殘差。
瀏覽器重現訓練補值、縮放、one-hot 和精確欄序，並逐筆與 Python 核對。
保留三題／方法／k 切換、預測、情境、CSV、深色與手機版；切題清除情境，不混合資料。
本機與 nested HTTP path 通過不等於公開 GitHub Pages 已更新；公開部署狀態以實際驗證為準。

### 本題實際圖表

{chr(10).join(image_blocks)}
""")
        manifests.append({'id':key,'run_sha256':digest,'pipeline_seconds':run['runtime_seconds'],
                          'diagnostics_seconds':diag['runtime_seconds'],'selected_method':best['method']})
    overview_table=table(['題目','樣本','CSV 欄位','原始輸入','完整編碼','目標','單位','Train / test'],overview)
    comparison_table=table(['題目','方法','encoded k','Inner CV RMSE','CV std','Test RMSE','Test R²'],comparisons)
    timing_table=table(['題目','Pipeline 秒','診斷／圖表秒','上述合計秒','cache namespace 已存在'],timings)
    report=f"""# AIoT HW2 — Multiple Linear Regression and Feature Selection: Student Performance, Appliances Energy and Car Price Prediction

## 摘要

本研究以三個真實資料問題實作 Multiple Linear Regression、Baseline 與五種 Feature Selection，
採 training-only nested CV、獨立 holdout、五假設診斷、95% 區間與三題靜態互動儀表板。
全部表格／圖檔來自實際 {mode} 產物，沒有虛構結果。
三題樣本與切分不同；RMSE 目標單位是 grade、Wh 與 USD，不能直接比較大小判定哪題模型較好。

## 緒論與問題定義

作業使用 14–39 個原始輸入，研究特徵選擇對泛化、可解釋性與回歸假設的影響。
以下依六階段 CRISP-DM 組織。

{overview_table}

## 資料來源與版本誠實性

指定學生 Kaggle archive 實際只有 Portuguese（649 筆），沒有將其改名為 Mathematics。
本專案改以已記錄來源的 UCI student-mat.csv，395 筆、33 欄：32 個非目標欄位再排除 G1/G2，得到 early-30。
能源為 Kaggle 真實副本，29 欄排除目標、日期與兩個隨機欄後得到 25 個輸入。汽車使用 hellbuoy Kaggle 真實檔，205 筆、26 欄，排除 ID／目標後保留 24 個原始候選欄位。
因此交付是三個真實驗證資料集，但不宣稱三個原檔都來自指定 Kaggle；此來源差異仍需教師接受。

## Multiple Linear Regression 與五種特徵選擇

最終 y=intercept+Xβ，以最小平方法擬合；LASSO 只用來選 support，不是最終 predictor。
Correlation 按 training 絕對 Pearson 排名；Mutual Information 估計互資訊，二元 indicator 視為 discrete；
RFE 沿標準化線性係數遞迴消除；RFECV 以完整 raw-input Pipeline CV 選 RFE 路徑上的 k，可能與 tuned RFE 相同；
LASSO 選非零 support，再以 LinearRegression 擬合。空 support 有明示的一欄 fallback。

FAST inner/outer 各 2 folds，k=1/5/10/20/all（依實際可行寬度），alpha fractions=0.01/0.1。
FULL inner=4、outer=5，探索全部可行 k 與五個 alpha fractions；本次沒有執行 FULL，不以 FAST 宣稱等價統計證據。
最多兩個 joblib workers，每個 native thread=1，outer loops 不再平行，cache 僅重用同一 training fold 的相同計算。

MAE 是平均絕對誤差；MSE 是平方誤差平均；RMSE 為 MSE 平方根、保留目標單位；R² 可為負。
Adjusted R² = 1−[SSE/(n−rank)]/[SST/(n−1)]，rank 為含截距的有效設計維度。

## 實際模型比較

{comparison_table}

CV std 是 fold RMSE 的標準差，不是平均的標準誤或信賴區間。推薦僅依 training inner CV，不依 test。
下面各題另列完整 train/test 指標。

## 實際執行時間

{timing_table}

Pipeline 時間含載入、nested/final CV、評估及模型／CSV 匯出，不含 Python 匯入與程序啟閉；可能含 cache hit。
診斷時間含指標、區間、檢定與 13 張圖。上述合計不含共用 dashboard export、browser QA 與文件產生；
共用步驟 wall time 另見驗收紀錄，不任意分攤到單題。

{chr(10).join(parts)}

## 三題比較、限制與改善

學生資料小、類別多、one-hot 與截距存在 aliasing，early-30 訊號有限，受切分與選擇不確定性影響；
能源雖大卻有序列相依與時段轉移，有效資訊量不等於獨立樣本數。三題 RMSE 不能跨單位直接比較。
後續先規劃 training-only FULL、能源時間 gap 敏感度與學生 group-aware 驗證；已觀察的 test 不應反覆用來調參。
非線性／不同誤差結構可作未來研究，本作業最終模型仍保留 LinearRegression。

## 結論

完成可追溯的三題線性迴歸、五種特徵選擇、完整診斷與互動介面。
保留負 R²、常態性違反、自相關等實際限制，不以測試資料重選方法，不宣稱傳統區間已校準或模型具有因果效果。

## AI 協助紀錄與待補附件

[AI_ASSISTANCE.md](AI_ASSISTANCE.md) 已收錄使用者提供的 NotebookLM 摘要；原始引用與使用日期、真實 ChatGPT 對話 PDF 及最終 PDF 匯出仍待補。
摘要屬研究對照，不把尚未實作的 RFE＋VIF、Random Forest 或 XGBoost 當作本專案結果。請補姓名／學號／課程與最終排版。

{(ROOT/'docs/NOTEBOOKLM_RESEARCH.md').read_text(encoding='utf-8')}

## 圖表索引與驗收

[FIGURE_INDEX.md](FIGURE_INDEX.md) 列出全部 39 張真實 PNG 的圖號、檔名、用途與章節。
[最終驗收 JSON](phase5/validation.json) 記錄命令、測試數與 runtime；各 run hash 見 report_manifest.json。

## 參考文獻
{REFS}
"""
    write(REPORTS/'CRISP_DM_REPORT.md',report)
    write(REPORTS/'FIGURE_INDEX.md','# 實際報告圖表索引\n\n'+table(
        ['圖號','Dataset ID','檔名','說明','建議章節','實際 PNG'],
        [[n,k,name,desc,section,f'[開啟]({path})'] for n,k,name,desc,section,path in figures]))
    with (REPORTS/'figure_index.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['figure_number','dataset_id','filename','description','section','path']);writer.writerows(figures)
    write(REPORTS/'AI_ASSISTANCE.md',f"""# AI Assistance Documentation — 摘要已提供，引用與附件待補

## A. ChatGPT Assistance Record

尚未附使用者真實 ChatGPT 對話 PDF。請匯出本作業實際對話，保留需求、修正、測試與限制；
檢查個資／憑證後附上，不將程式執行紀錄冒充對話 PDF。

- 實際 PDF 檔名／位置：**［待填］**
- 日期／對話範圍：**［待填］**
- AI 使用方式與人工核對：**［待填］**

## B. NotebookLM Research Summary

狀態：已收到使用者提供的 NotebookLM 研究摘要，超過 100 個中文字。原文及對照如下；來源引用與使用日期仍待補。

{(ROOT/'docs/NOTEBOOKLM_RESEARCH.md').read_text(encoding='utf-8')}

## C. 最終 PDF 報告

使用 README.md 或 CRISP_DM_REPORT.md 為稿，補姓名／學號／課程、頁碼及附件，再用 Markdown 編輯器或 Pandoc 匯出 PDF。
人工檢查中文字型、公式、39 張 PNG、跨頁表格及連結。此處不宣稱 PDF 已完成。
""")

    write(REPORTS/'phase5/report_manifest.json',json.dumps(
        {'mode':mode,'full_executed_in_phase5':False,'figure_count':len(figures),'datasets':manifests},indent=2))
    return overview_table,comparison_table,timing_table


def readme_evidence(mode):
    """Read report-only evidence; no fitting, inference changes, or external AI calls."""
    labels={'student_performance':'學生','appliances_energy':'能源','car_price':'汽車'}
    names={'linearity':'線性','independence':'獨立性','homoscedasticity':'等變異','normality':'常態性','multicollinearity':'共線性'}
    metric_rows,diagnostic_rows=[],[]
    def ptext(value):
        return '不適用' if value is None else '低於浮點精度' if value==0 else format(value,'.2e') if value<.0001 else f(value)
    for key in OFFICIAL_DATASETS:
        run=read(ROOT/'results'/key/mode/'run.json')
        method=run['best_model']['method']
        with (REPORTS/'tables'/key/mode/'evaluation_metrics.csv').open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):
                if row['method']!=method:continue
                metric_rows.append([labels[key],method,row['split'],row['fitted_design_rank'],
                    *[f(float(row[n])) if row[n] else '不適用' for n in ['mae','mse','rmse','r2','adjusted_r2']]])
        diag=read(REPORTS/'tables'/key/mode/'diagnostics.json')['models'][method]
        for a in diag['assumptions']:
            t=a['statistic'];kind=a['assumption']
            if kind=='linearity':stat='分箱殘差均值跨度 '+f(t.get('bin_mean_span_residual_sd'),3)+' SD'
            elif kind=='independence':stat='DW='+f(t.get('durbin_watson'),3)+'；lag-1 ACF='+f(t.get('acf',[None,None])[1],3)
            elif kind=='homoscedasticity':stat='LM='+f(t.get('lm_statistic'))+'；p='+ptext(t.get('lm_pvalue'))
            elif kind=='normality':stat=str(t.get('test',''))+'；p='+ptext(t.get('pvalue'))+'；偏態='+f(t.get('skewness'),3)
            else:stat='rank='+str(t.get('rank'))+'/'+str(t.get('columns_with_intercept'))+'；完美共線欄數='+str(t.get('perfectly_collinear_columns'))
            status='可能違反假設' if a['possible_violation'] is True else '未觸發篩檢，非證實成立' if a['possible_violation'] is False else '描述性觀察，非證實成立'
            diagnostic_rows.append([labels[key],names[kind],stat,status])
    validation_path=REPORTS/'phase5/validation.json'
    validation=read(validation_path) if validation_path.is_file() else {}
    if validation.get('passed'):
        unit=validation['unit_tests'];browser=validation['browser']
        verification=('最近一輪完整驗收：**'+str(unit['passed'])+' 項單元測試通過、'+str(unit['failures'])+' 失敗、'+str(unit['skipped'])+' 跳過**；後端、診斷及瀏覽器整合均通過。瀏覽器核對 '+
                      format(browser['counts']['models'],',')+' 個候選模型、'+format(browser['counts']['predictions'],',')+' 筆預測及 '+format(browser['counts']['interval_rows'],',')+
                      ' 筆區間，沒有 JavaScript 執行錯誤；另檢查 1440／768／390／320 px、CSV、情境隔離及深／淺色。完整命令、時間與資產雜湊見 [驗收 JSON](reports/phase5/validation.json)。本次 README 更新重用這些實驗產物，未重新訓練或新增模型比較。')
    else:
        verification='尚無通過的完整驗收紀錄，請執行 validate_project.py 並檢查 reports/phase5/validation.json。'
    return {'{{SELECTED_METRICS}}':table(['題目','選定方法','資料','有效 rank','MAE','MSE','RMSE','R²','Adjusted R²'],metric_rows),
            '{{DIAGNOSTICS}}':table(['題目','假設','實際統計','判讀'],diagnostic_rows),
            '{{NOTEBOOKLM}}':(ROOT/'docs/NOTEBOOKLM_RESEARCH.md').read_text(encoding='utf-8'),
            '{{VALIDATION}}':verification}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['FAST','FULL'],default='FAST',type=str.upper)
    args=parser.parse_args()
    overview,comparison,timing=prepare(args.mode)
    template=(ROOT/'docs/README.template.md').read_text(encoding='utf-8')
    for key,value in readme_evidence(args.mode).items():template=template.replace(key,value)
    write(ROOT/'README.md',template.replace('{{OVERVIEW}}',overview).replace('{{RESULTS}}',comparison).replace('{{TIMING}}',timing).replace('{{MODE}}',args.mode))
    print('Generated actual CRISP-DM report, figure index and explicit AI placeholders')
if __name__=='__main__':
    main()
