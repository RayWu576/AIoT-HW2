(() => {
'use strict';
const BUNDLE=window.DASHBOARD_DATA,$=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
if(!BUNDLE||BUNDLE.schemaVersion!==5||!window.DashboardPrediction||Object.keys(BUNDLE.datasets||{}).join(',')!=='student_performance,appliances_energy,car_price'){
 document.querySelector('main').innerHTML='<section class="panel"><h2>Dashboard data unavailable</h2><p>Export both problems with <code>python dashboard_export.py --mode FAST</code>, then reload. The data and app versions must match.</p></section>';return;
}
const methods=['Baseline','Correlation','Mutual Information','RFE','RFECV','LASSO'],colors=['#94a3b8','#5585ef','#09ad87','#e7a43c','#9c68db','#e16a87'];
const notes=['All encoded inputs; final LinearRegression.','Absolute training Pearson ranking.','Training mutual information; binary indicators treated as discrete.','Scaled linear recursive elimination.','Full-Pipeline CV along the RFE elimination path.','L1 nonzero support only; the final predictor remains LinearRegression.'];
let datasetId='student_performance',D=BUNDLE.datasets[datasetId];
const s={modelId:D.defaultModelId,compare:true,values:{},sort:'abs',page:0,scenarios:[],metricSplit:'test'};
const num=(v,d=4)=>v===null||v===undefined||!Number.isFinite(Number(v))?'\u2014':Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const unit=v=>num(v,2)+' '+D.metadata.unit,pv=v=>v===null?'\u2014':v<.0001?'<0.0001':num(v);
const model=()=>D.models.find(m=>m.id===s.modelId),isBest=m=>m&&m.id===D.defaultModelId;
const storage={get:k=>{try{return localStorage.getItem(k)}catch{return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch{}}};
if(storage.get('startup-theme')==='dark')document.body.classList.add('dark');
$('theme').textContent='Theme';$('theme').setAttribute('aria-label','Toggle dark mode');
$('theme').onclick=()=>{document.body.classList.toggle('dark');storage.set('startup-theme',document.body.classList.contains('dark')?'dark':'light');};
function toast(t){$('toast').textContent=t;$('toast').classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.remove('show'),2500);}
function choose(method,k,id){s.modelId=id||D.models.find(m=>m.method===method&&m.k===k)?.id||D.models.find(m=>m.method===method&&m.isOfficial)?.id||D.models.find(m=>m.method===method)?.id;s.page=0;render();}
function controls(){
 const m=model();$('method-select').innerHTML=methods.map(n=>`<option ${n===m.method?'selected':''}>${n}</option>`).join('');
 $('methods').innerHTML=methods.map((n,i)=>`<button class="method-button ${n===m.method?'active':''}" data-method="${n}" aria-pressed="${n===m.method}"><span class="method-icon">${i===0?'ALL':i}</span><strong>${n}</strong><small>${i===0?'All features':notes[i]}</small></button>`).join('');
 $('methods').querySelectorAll('button').forEach(b=>b.onclick=()=>choose(b.dataset.method,m.k));
 $('counts').innerHTML=D.models.filter(x=>x.method===m.method).sort((a,b)=>a.k-b.k).map(x=>`<button data-k="${x.k}" data-model="${x.id}" aria-pressed="${x.id===m.id}" class="${x.id===m.id?'active':''}">k = ${x.k}${x.isOfficial?'<small>CV</small>':''}</button>`).join('');
 $('counts').querySelectorAll('button').forEach(b=>b.onclick=()=>choose(m.method,+b.dataset.k,b.dataset.model));
 $('method-note').textContent=notes[methods.indexOf(m.method)]+` k counts encoded columns; original input count is ${D.metadata.originalFeatureCount}. Only recorded CV candidates are available. LASSO k is final-refit support; fold supports vary. `+(isBest(m)?'Training-CV recommended model.':m.isOfficial?'Training-CV final model for this method.':'Exploratory recorded candidate; not the recommended model.');
 $('compare').checked=s.compare;
}
function metrics(m){
 const v=m[s.metricSplit],label=s.metricSplit==='test'?'TEST':'TRAIN';
 const items=[['INNER CV RMSE',unit(m.cvRmse),'Tuning score; std '+num(m.cvStd,3)],[label+' RMSE',unit(v.rmse),'Separate from CV'],[label+' R2',num(v.r2),'Can be negative'],[label+' ADJUSTED R2',num(v.adjusted_r2),'Effective fitted rank '+m.rank],[label+' MAE',unit(v.mae),'Mean absolute error'],[label+' MSE',num(v.mse,2),D.metadata.unit+' squared']];
 $('metrics').innerHTML=items.map(([title,value,note])=>`<article class="metric"><div class="metric-label">${title}</div><strong>${value}</strong><p>${note}</p></article>`).join('');
 $('active-features').innerHTML=m.features.map(f=>`<span class="feature-chip" title="${esc(f)}">${esc(f)}</span>`).join('');
 $('insight').textContent=`${m.k} encoded columns / ${m.originalFeatures.length} represented original fields; effective rank ${m.rank}.`;
 $('equation').innerHTML=`${D.metadata.target} = <b>${num(m.intercept,3)}</b> `+m.features.map((f,i)=>`${m.weights[i]>=0?'+':'\u2212'} <b>${num(Math.abs(m.weights[i]),5)}</b> (${esc(f)})`).join(' ');
}
function chart(id,metric){
 const current=model(),visible=s.compare?methods:[current.method],points=D.models.filter(m=>visible.includes(m.method)&&m[metric]!==null);
 if(!points.length){$(id).textContent='No recorded CV scores are available.';return;}
 const values=points.map(m=>m[metric]),min=Math.min(...values),max=Math.max(...values),pad=(max-min)*.15||1,lo=min-pad,hi=max+pad,W=560,H=270,L=60,R=18,T=18,B=40;
 const largest=Math.max(...D.models.map(m=>m.k)),x=k=>L+(k-1)*(W-L-R)/Math.max(1,largest-1),y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
 const available=[...new Set(D.models.map(m=>m.k))].sort((a,b)=>a-b),ticks=available.length<=8?available:available.filter((_,i)=>i===0||i===available.length-1||i%Math.ceil(available.length/6)===0);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training validation ${metric} by encoded feature count">`;
 for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;svg+=`<line class="grid-line" x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}"/><text x="${L-8}" y="${y(v)+3}" text-anchor="end">${num(v,metric==='cvRmse'?1:2)}</text>`;}
 ticks.forEach(k=>svg+=`<text x="${x(k)}" y="${H-20}" text-anchor="middle">${k}</text>`);
 visible.forEach(method=>{const series=points.filter(m=>m.method===method).sort((a,b)=>a.k-b.k),color=colors[methods.indexOf(method)];svg+=`<path d="${series.map((m,i)=>`${i?'L':'M'}${x(m.k)} ${y(m[metric])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${method===current.method?3:1.6}" ${method==='RFECV'?'stroke-dasharray="5 4"':''}/>`;
 series.forEach(m=>svg+=`<circle cx="${x(m.k)}" cy="${y(m[metric])}" r="${m.id===current.id?6:3.8}" fill="${color}" stroke="var(--surface)" stroke-width="1.5" tabindex="0" role="button" data-model="${m.id}" aria-label="${method}, k=${m.k}, ${metric}=${num(m[metric],4)}"><title>${method}, k=${m.k}: ${num(m[metric],4)}</title></circle>`);
 });
 svg+=`<text x="${W/2}" y="${H-2}" text-anchor="middle">Encoded count k (all at final training width)</text></svg>`;
 $(id).innerHTML='<div class="legend">'+visible.map(n=>`<button data-method="${n}"><i style="background:${colors[methods.indexOf(n)]}"></i>${n}</button>`).join('')+'</div>'+svg+'<div class="chart-tooltip">Click a recorded candidate. No test scores choose the recommendation.</div>';
 $(id).querySelectorAll('.legend button').forEach(b=>b.onclick=()=>choose(b.dataset.method,current.k));
 $(id).querySelectorAll('circle').forEach(c=>{const activate=()=>choose(null,null,c.dataset.model);c.onclick=activate;c.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};});
}
function coefficients(m){
 $('coefficients').innerHTML=m.coefficients.map(c=>`<tr><td title="${esc(c.name)}">${esc(c.name)}${!c.identifiable?'<small> aliased</small>':''}</td><td><strong>${num(c.estimate)}</strong></td><td>${num(c.se)}</td><td>${num(c.t,2)}</td><td>${pv(c.p)}</td><td>[${num(c.low,3)}, ${num(c.high,3)}]</td><td>${c.vifStatus==='perfect_collinearity'?'Infinity (aliased)':num(c.vif,2)}</td></tr>`).join('');
 $('model-statistics').textContent=`Training F=${num(m.f,2)}, p ${pv(m.fP)}; rank=${m.rank}/${m.k+1}, residual df=${m.uncertainty.residualDf}. Unidentifiable coefficients have no individual SE/CI. Statistics are exploratory after selection.`;
 const effects=m.features.map((f,i)=>Math.abs(m.weights[i]*(D.ranges[f].max-D.ranges[f].min))),maximum=Math.max(...effects,1);
 $('coefficient-bars').innerHTML='<div class="eyebrow purple">CHANGE OVER OBSERVED ENCODED TRAINING RANGE</div>'+m.features.map((f,i)=>`<div class="bar-row"><span title="${esc(f)}">${esc(f)}</span><div class="bar-track"><div class="bar-fill" style="width:${effects[i]/maximum*100}%"></div></div><span>${num(effects[i],2)}</span></div>`).join('');
}
function inputs(){
 $('inputs').innerHTML=D.inputs.map((f,i)=>f.kind==='categorical'?`<div><label class="field-label" for="category-${i}">${esc(f.label)}</label><select id="category-${i}" aria-label="${esc(f.label)}">${f.options.map(v=>`<option value="${esc(v)}" ${s.values[f.name]===v?'selected':''}>${esc(v)}</option>`).join('')}</select></div>`:`<div><label class="field-label" for="number-${i}">${esc(f.label)} ${esc(f.unit)}</label><div class="input-pair"><input id="number-${i}" type="number" step="${f.step}" ${f.domainMin===null?'':`min="${f.domainMin}"`} ${f.domainMax===null?'':`max="${f.domainMax}"`} value="${s.values[f.name]??''}" aria-label="${esc(f.label)}"><input id="range-${i}" type="range" min="${Math.min(f.min,s.values[f.name]??f.min)}" max="${Math.max(f.max,s.values[f.name]??f.max)}" step="${f.step}" value="${s.values[f.name]??f.default}" aria-label="${esc(f.label)} slider"></div><small class="input-range">Observed train range: ${num(f.min,2)} - ${num(f.max,2)} ${esc(f.unit)}</small></div>`).join('');
 D.inputs.forEach((f,i)=>{if(f.kind==='categorical'){$('category-'+i).onchange=e=>{s.values[f.name]=e.target.value;predict();};return;}
  const n=$('number-'+i),r=$('range-'+i);n.oninput=()=>{s.values[f.name]=n.value===''?null:n.valueAsNumber;if(Number.isFinite(s.values[f.name])){r.min=Math.min(+r.min,s.values[f.name]);r.max=Math.max(+r.max,s.values[f.name]);r.value=s.values[f.name];}predict();};r.oninput=()=>{s.values[f.name]=+r.value;n.value=r.value;predict();};
 });predict();
}
function predict(){
 const m=model(),validation=DashboardPrediction.validate(D,s.values);$('input-errors').textContent=validation.errors.join(' ');$('save-scenario').disabled=validation.errors.length>0;
 if(validation.errors.length){$('profit').textContent='\u2014';$('prediction-intervals').textContent='Complete the inputs to calculate intervals.';$('contributions').textContent='';$('extrapolation').textContent='';return null;}
 try{
  const result=DashboardPrediction.intervals(D,m,s.values);$('profit').textContent=unit(result.prediction);
  $('prediction-intervals').innerHTML=result.estimable?`<strong>Approximate pointwise 95% intervals</strong><p>Conditional mean CI: [${unit(result.meanLower)}, ${unit(result.meanUpper)}]</p><p>Individual observation PI: [${unit(result.piLower)}, ${unit(result.piUpper)}]</p><p>${result.robustType} asymptotic mean CI: [${unit(result.robustMeanLower)}, ${unit(result.robustMeanUpper)}]</p><small>${esc(m.uncertainty.limitations)}</small>`:'Intervals not estimable: this input lies outside the training affine span. Point prediction uses the saved coefficient convention.';
  if(result.leverage>m.uncertainty.maxTrainingLeverage+1e-8)validation.warnings.push('Unusual feature combination: leverage exceeds the training maximum.');
  if(!result.estimable)validation.warnings.push('Conditional mean is not identifiable for this combination.');
  $('extrapolation').textContent=validation.warnings.join(' ')+' '+D.metadata.targetNote;
  const encoded=DashboardPrediction.encode(D.preprocessing,s.values);
  $('contributions').textContent=m.features.map((f,i)=>`${f}: ${num(m.weights[i]*encoded[f],3)}`).join(' / ');
  return result;
 }catch(error){$('input-errors').textContent='Prediction could not be computed: '+error.message;$('profit').textContent='\u2014';$('prediction-intervals').textContent='No interval available.';$('save-scenario').disabled=true;return null;}
}

function scenarios(){
 $('scenarios').innerHTML=s.scenarios.map((r,i)=>`<article class="scenario"><strong>${esc(r.method)} / k=${r.k}</strong><p>${unit(r.prediction)}</p><small>95% PI: ${r.estimable?'['+unit(r.piLower)+', '+unit(r.piUpper)+']':'Not estimable'}</small><details><summary>Inputs</summary><pre>${esc(JSON.stringify(r.values,null,2))}</pre></details><button data-remove="${i}">Remove</button></article>`).join('');
 $('scenarios').querySelectorAll('button').forEach(b=>b.onclick=()=>{s.scenarios.splice(+b.dataset.remove,1);scenarios();});
}
function rows(m){return D.testRows.map((r,i)=>({...r,predicted:m.testPredictions[i],residual:r.actual-m.testPredictions[i],absolute:Math.abs(r.actual-m.testPredictions[i]),percent:r.actual===0?null:100*Math.abs((r.actual-m.testPredictions[i])/r.actual)}));}
function scatter(id,points,xlabel,ylabel,reference,line){
 const W=600,H=285,L=66,R=18,T=18,B=45;
 let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);if(reference==='identity'){const all=xs.concat(ys);xs=all;ys=all;}if(reference==='zero')ys=ys.concat([0]);
 const extent=a=>{const lo=Math.min(...a),hi=Math.max(...a),pad=(hi-lo)*.08||1;return[lo-pad,hi+pad];},[xmin,xmax]=extent(xs),[ymin,ymax]=extent(ys),x=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R),y=v=>H-B-(v-ymin)/(ymax-ymin)*(H-T-B);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylabel)} versus ${esc(xlabel)}">`;
 for(let i=0;i<5;i++){const xv=xmin+(xmax-xmin)*i/4,yv=ymin+(ymax-ymin)*i/4;svg+=`<line class="grid-line" x1="${L}" x2="${W-R}" y1="${y(yv)}" y2="${y(yv)}"/><text x="${L-7}" y="${y(yv)+3}" text-anchor="end">${num(yv,1)}</text><text x="${x(xv)}" y="${H-B+18}" text-anchor="middle">${num(xv,1)}</text>`;}
 if(reference==='identity')svg+=`<path d="M${x(xmin)} ${y(xmin)} L${x(xmax)} ${y(xmax)}" stroke="#e7a43c" fill="none"/>`;
 if(reference==='zero')svg+=`<path d="M${L} ${y(0)} L${W-R} ${y(0)}" stroke="#e7a43c" fill="none"/>`;
 if(line)svg+=`<path d="${line.map((p,i)=>`${i?'L':'M'}${x(p[0])} ${y(p[1])}`).join(' ')}" stroke="#e16a87" stroke-width="2" fill="none"/>`;
 points.forEach(p=>svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="2.3" fill="#5585ef" opacity=".45"><title>${num(p[0],3)}, ${num(p[1],3)}</title></circle>`);
 svg+=`<text x="${W/2}" y="${H-3}" text-anchor="middle">${esc(xlabel)}</text><text transform="translate(13 ${H/2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text></svg>`;$(id).innerHTML=svg;
}

function diagnosticCard(a,index,m){
 const t=a.statistic||{},known=v=>v!==null&&v!==undefined&&Number.isFinite(Number(v));
 const value=(v,d=3)=>known(v)?num(v,d):'無法估計';
 const probability=v=>!known(v)?'無法估計':Number(v)===0?'低於浮點精度':Number(v)<.0001?Number(v).toExponential(2):num(v,4);
 const fields=[],add=(label,v)=>fields.push([label,v]);
 const names={linearity:['線性關係','Linearity'],independence:['殘差獨立性','Independence'],homoscedasticity:['等變異性','Homoscedasticity'],normality:['殘差常態性','Normality'],multicollinearity:['多重共線性','Multicollinearity']};
 const [title,english]=names[a.assumption]||[a.assumption,'Regression diagnostic'];
 const flag=a.possible_violation===true,status=flag?'warn':a.possible_violation===false?'screened':'descriptive';
 const badge=flag?'可能違反假設':status==='screened'?'未觸發篩檢警示':'描述性觀察';
 let meaning='',caution='';
 switch(a.assumption){
 case 'linearity':
  add('分箱數',value(t.bin_count,0));add('殘差均值跨度',known(t.bin_mean_span_residual_sd)?value(t.bin_mean_span_residual_sd)+' SD':'無法估計');
  add('分箱殘差均值範圍',known(t.minimum_bin_mean)&&known(t.maximum_bin_mean)?value(t.minimum_bin_mean,2)+' ～ '+value(t.maximum_bin_mean,2)+' '+D.metadata.unit:'無法估計');
  meaning=known(t.bin_mean_span_residual_sd)?flag?'殘差分箱均值的變化較大，可能存在未捕捉的非線性趨勢。':'殘差分箱均值未觸發目前的趨勢篩檢，請搭配上方殘差圖判讀。':'不同擬合值或殘差變異不足，目前無法判讀趨勢。';
  caution='跨度超過 0.5 SD 是視覺篩檢慣例，不是顯著性檢定；未觸發也不能證明線性成立。';break;
 case 'independence':{
  add('Durbin–Watson',value(t.durbin_watson));add('Lag-1 ACF',value(t.acf?.[1]));
  const ps=(t.ljung_box||[]).map(r=>r.pvalue).filter(known);
  add('Ljung–Box 最小 p 值',ps.length?probability(Math.min(...ps)):'未進行時間序列檢定');
  meaning=datasetId==='appliances_energy'?(flag?'檢定提示殘差存在時間相依，前後觀測的誤差可能互相關聯。':'目前未標記時間相依違反，仍需搭配 ACF 與時間序列檢定判讀。'):datasetId==='car_price'?'汽車原始列序不是時間；DW／ACF 只作描述，不能據此證明車輛觀測獨立。':'學生來源行序不是時間；這裡的 DW／ACF 只作描述，無法據此證明觀測獨立。';
  caution=datasetId==='appliances_energy'?'時間相依會限制傳統 OLS 推論；Ljung–Box 是選模後的探索性檢定，含多個相關 lag。':datasetId==='car_price'?'同廠牌、車型或其他群組的相依性仍可能存在；小樣本會限制統計推論。':'同校、家庭或其他群組的相依性仍可能存在，不能只看 DW 是否接近 2。';break;
 }
 case 'homoscedasticity':
  add('LM 統計量',value(t.lm_statistic));add('Breusch–Pagan p 值',probability(t.lm_pvalue));add('輔助迴歸自由度',value(t.auxiliary_df,0));
  meaning=known(t.lm_pvalue)?flag?'有反對固定誤差變異的證據：不同輸入情況下，誤差的波動可能不同。':'目前未拒絕等變異假設，仍需搭配殘差散布判讀。':'目前無法估計等變異檢定，請查看詳細資訊與殘差圖。';
  caution='未拒絕不等於證明等變異；時間相依與特徵選擇會影響檢定 p 值的解釋。';break;
 case 'normality':
  add('檢定統計量',value(t.statistic,2));add('p 值',probability(t.pvalue));add('偏態 Skewness',value(t.skewness));add('Pearson 峰度',value(t.pearson_kurtosis));
  meaning=known(t.pvalue)?flag?'常態性檢定拒絕常態假設；請搭配直方圖及 Q-Q 圖觀察偏態與尾部。':'目前未拒絕常態假設；這不代表殘差已被證明服從常態。':'殘差變異或樣本資訊不足，無法進行常態性判讀。';
  caution='大樣本檢定對微小偏離也很敏感；「低於浮點精度」表示數值下溢，不是機率真的等於零。';break;
 case 'multicollinearity':
  add('有效 rank / 含截距欄數',known(t.rank)&&known(t.columns_with_intercept)?value(t.rank,0)+' / '+value(t.columns_with_intercept,0):'無法估計');
  add('Condition number',value(t.condition_number));add('完美共線欄數',value(t.perfectly_collinear_columns,0));add('有限 VIF > 10 欄數',value(t.vif_above_10,0));
  meaning=flag?'設計矩陣可能有別名欄位或高度共線性；個別係數不一定能單獨解釋。':m.k===1?'此模型只有一個選中輸入，沒有多個輸入互相共線的問題；仍需檢查常數欄與有效 rank。':'目前未觸發 rank／VIF 篩檢警示，仍須留意特徵間的關聯。';
  caution='完整 one-hot 與截距可能造成 aliasing。VIF 的空值可能代表無限大或不可定義，不能視為 0；門檻不是通用定律。';break;
 default:meaning=a.interpretation||'請查看詳細統計資料。';caution=a.limitations||'';
 }
 return '<article class="diagnostic-card diagnostic-card--'+status+'" data-assumption="'+esc(a.assumption)+'" aria-labelledby="diagnostic-title-'+index+'">'+
 '<div class="diagnostic-card-header"><div class="diagnostic-name"><span class="diagnostic-number" aria-hidden="true">'+String(index+1).padStart(2,'0')+'</span><div><h4 id="diagnostic-title-'+index+'">'+esc(title)+'</h4><span class="diagnostic-english">'+esc(english)+'</span></div></div><span class="diagnostic-badge"><span aria-hidden="true">'+(flag?'!':'•')+'</span>'+badge+'</span></div>'+
 '<p class="diagnostic-meaning">'+esc(meaning)+'</p><dl class="diagnostic-metrics">'+fields.map(([label,v])=>'<div><dt>'+esc(label)+'</dt><dd>'+esc(v)+'</dd></div>').join('')+'</dl>'+
 '<p class="diagnostic-caution">'+esc(caution)+'</p><details class="diagnostic-details"><summary>檢定方法與完整統計<span aria-hidden="true">＋</span></summary><div class="diagnostic-detail-body"><p><strong>方法</strong> '+esc(a.method)+'</p><p>'+esc(a.interpretation)+'</p><p>'+esc(a.limitations)+'</p><pre aria-label="原始統計 JSON">'+esc(JSON.stringify(a.statistic,null,2))+'</pre></div></details></article>';
}
function diagnosticSummary(m){
 const d=m.diagnostics,flags=m.assumptions.filter(a=>a.possible_violation===true).length;
 return '<div class="diagnostic-overview"><div><span class="eyebrow purple">REGRESSION ASSUMPTIONS</span><h3>回歸假設檢查</h3><p>先看重點，再展開細節。以下為訓練資料的探索性診斷，未觸發警示不等於假設成立。</p></div><span class="diagnostic-overview-count">'+flags+' / '+m.assumptions.length+'<small>項需留意</small></span></div>'+
 '<div class="diagnostic-cards">'+m.assumptions.map((a,i)=>diagnosticCard(a,i,m)).join('')+'</div>'+
 '<p class="diagnostic-footnote">圖中顯示 '+d.displayedTrainingPoints.toLocaleString('en-US')+' / '+d.trainingSize.toLocaleString('en-US')+' 筆訓練觀測（含殘差極值），Q-Q 最多顯示 201 個分位點；檢定使用完整訓練樣本。這些診斷不代表因果關係，也未校正選模不確定性。</p>';
}

function diagnosticPlots(m){
 const d=m.diagnostics,u=D.metadata.unit;
 scatter('residual-fit-chart',d.residualScatter,'Fitted ('+u+')','Training residual ('+u+')','zero',d.binnedResiduals.map(b=>[b.fitted_mean,b.residual_mean]));
 const q=d.qq,ql=d.qqLine;scatter('qq-chart',q,'Normal theoretical quantile','Ordered training residual ('+u+')',null,[[q[0][0],ql.intercept+ql.slope*q[0][0]],[q.at(-1)[0],ql.intercept+ql.slope*q.at(-1)[0]]]);
 const h=d.histogram,mid=h.edges.slice(0,-1).map((v,i)=>(v+h.edges[i+1])/2),pts=mid.map((v,i)=>[v,h.trainDensity[i]]);scatter('residual-distribution-chart',pts,'Residual ('+u+')','Probability density',null,mid.map((v,i)=>[v,h.testDensity[i]]));
 $('residual-distribution-chart').insertAdjacentHTML('beforeend','<p class="muted">Blue: training density; pink: held-out test density. Equal-width histogram bins.</p>');
 scatter('acf-chart',d.acf.map((v,i)=>[i,v]),'Lag (ordered training rows)','Residual ACF','zero');
 $('diagnostic-interpretation').innerHTML=diagnosticSummary(m);
}
function residuals(m){
 let r=rows(m);r.sort(s.sort==='id'?(a,b)=>a.id-b.id:s.sort==='percent'?(a,b)=>(b.percent??-1)-(a.percent??-1):(a,b)=>b.absolute-a.absolute);
 const page=r.slice(s.page*20,s.page*20+20);$('residuals').innerHTML=page.map(v=>`<tr><td>${v.id}${v.date?'<small>'+esc(v.date)+'</small>':''}</td><td>${num(v.actual,2)}</td><td>${num(v.predicted,2)}</td><td>${num(v.residual,2)}</td><td>${num(v.absolute,2)}</td><td>${v.percent===null?'N/A (zero target)':num(v.percent,1)+'%'}</td><td>${v.absolute>m.test.rmse?'Above RMSE':'Within RMSE'}</td></tr>`).join('');
 const pages=Math.ceil(r.length/20);$('residual-pager').innerHTML=`<button id="prev-page" ${s.page===0?'disabled':''}>Previous</button><span>${s.page+1} / ${pages} (${r.length} test rows)</span><button id="next-page" ${s.page===pages-1?'disabled':''}>Next</button>`;
 $('prev-page').onclick=()=>{s.page--;residuals(m);};$('next-page').onclick=()=>{s.page++;residuals(m);};
 const sample=rows(m).filter((_,i)=>i%Math.max(1,Math.floor(D.testSize/400))===0);scatter('scatter',sample.map(v=>[v.actual,v.predicted]),'Actual ('+D.metadata.unit+')','Predicted ('+D.metadata.unit+')','identity');
 $('residual-description').textContent=`${D.testSize} held-out observations; scatter displays ${sample.length}. Residual = actual - predicted. Units: ${D.metadata.unit}.`;
}
function render(){const m=model();controls();metrics(m);chart('rmse-chart','cvRmse');chart('r2-chart','validationR2');coefficients(m);inputs();residuals(m);diagnosticPlots(m);scenarios();}
function setupDataset(){
 D=BUNDLE.datasets[datasetId];Object.assign(s,{modelId:D.defaultModelId,values:Object.fromEntries(D.inputs.map(f=>[f.name,f.default])),page:0,sort:'abs',scenarios:[],metricSplit:'test'});
 const d=D.metadata;$('dataset-select').innerHTML=Object.entries(BUNDLE.datasets).map(([id,v])=>`<option value="${id}" ${id===datasetId?'selected':''}>${esc(v.metadata.problem)}</option>`).join('');
 $('dataset-summary').textContent=`${d.rawSize} observations / ${d.originalFeatureCount} original predictors / ${d.target} (${d.unit})`;$('dataset-intro').textContent=d.description;$('breadcrumb').textContent=d.problem;$('dataset-tag').textContent=d.originalFeatureCount+' original inputs';$('split-tag').textContent=`Train ${D.trainSize} / Test ${D.testSize}`;$('rmse-unit').textContent=d.unit;$('predictor-title').textContent=d.title;$('prediction-target').textContent=d.target+' ('+d.unit+')';$('metric-split').value='test';$('sort').value='abs';
 $('overview-grid').innerHTML=[['Problem',d.title],['Observations',d.rawSize],['Original predictors',d.originalFeatureCount],['Encoded full matrix',d.encodedFeatureCount],['Target / unit',d.target+' / '+d.unit],['Evaluation',d.splitDescription],['Run mode',D.mode+(D.mode==='FAST'?' (development evidence)':' (full evaluation)')],['Train / test',D.trainSize+' / '+D.testSize]].map(([a,b])=>`<div><small>${esc(a)}</small><strong>${esc(b)}</strong></div>`).join('');
 $('ranking').innerHTML=D.comparison.map((r,i)=>`<tr><td>${i+1}</td><td><button data-model="${r.modelId}">${esc(r.method)}</button></td><td>${r.k} / ${r.rank}</td><td>${num(r.cvRmse)}</td><td>${num(r.cvStd)}</td><td>${num(r.nestedCvRmse)} +/- ${num(r.nestedCvStd)}</td><td>${num(r.train.rmse)}</td><td>${num(r.test.rmse)}</td><td>${num(r.test.r2)}</td><td>${num(r.test.mae)}</td><td>${num(r.test.mse)}</td></tr>`).join('');$('ranking').querySelectorAll('button').forEach(b=>b.onclick=()=>choose(null,null,b.dataset.model));
 $('dataset-methodology').textContent=d.splitDescription+'. All preprocessing and selection were fitted inside training folds. Final candidates refit on training rows only. '+d.targetNote+' '+d.sourceNote;
 $('dataset-sources').innerHTML='<p><a href="'+esc(d.kaggleSource)+'" target="_blank" rel="noopener noreferrer">Kaggle source</a>'+(d.source===d.kaggleSource?'':' / <a href="'+esc(d.source)+'" target="_blank" rel="noopener noreferrer">Original UCI source</a>')+'</p>';
 $('feature-definitions').innerHTML=D.inputs.map(f=>`<tr><td>${esc(f.name)}</td><td>${esc(f.label)}</td><td>${f.kind} / ${esc(f.unit||'category')}</td></tr>`).join('');
 $('interval-methodology').textContent=model().uncertainty.limitations;$('generated').textContent=`Model run: ${D.generatedAt}; dataset ${datasetId}; ${D.mode}; source run SHA256 ${D.sourceRunSha256}.`;
 render();
}
$('dataset-select').onchange=e=>{datasetId=e.target.value;setupDataset();};$('method-select').onchange=e=>choose(e.target.value,model().k);$('metric-split').onchange=e=>{s.metricSplit=e.target.value;metrics(model());};$('compare').onchange=e=>{s.compare=e.target.checked;chart('rmse-chart','cvRmse');chart('r2-chart','validationR2');};$('sort').onchange=e=>{s.sort=e.target.value;s.page=0;residuals(model());};$('reset').onclick=setupDataset;
$('save-scenario').onclick=()=>{const r=predict();if(!r)return;if(s.scenarios.length>=5)s.scenarios.shift();s.scenarios.push({...r,method:model().method,k:model().k,values:{...s.values}});scenarios();toast('Scenario saved for this problem.');};
$('download').onclick=()=>{
 const m=model(),interval=predict(),output=[['dataset_id',datasetId],['target',D.metadata.target],['unit',D.metadata.unit],['mode',D.mode],['model',m.method],['encoded_k',m.k],['original_features',D.metadata.originalFeatureCount],['cv_rmse',m.cvRmse],['cv_std',m.cvStd],['source_run_sha256',D.sourceRunSha256],[],['scope','mae','mse','rmse','r2','adjusted_r2']];
 for(const scope of ['train','test'])output.push([scope,...['mae','mse','rmse','r2','adjusted_r2'].map(k=>m[scope][k])]);
 output.push([],['input','value'],...Object.entries(s.values),[],['prediction_interval',JSON.stringify(interval)],[],['coefficient','estimate','se','p','low','high'],...m.coefficients.map(c=>[c.name,c.estimate,c.se,c.p,c.low,c.high]),[],['id','date','actual','predicted','residual','absolute_error','percent_error'],...rows(m).map(r=>[r.id,r.date||'',r.actual,r.predicted,r.residual,r.absolute,r.percent]));
 const csv='\ufeff'+output.map(row=>row.map(v=>'"'+String(v??'').replace(/"/g,'""')+'"').join(',')).join('\r\n'),blob=new Blob([csv],{type:'text/csv;charset=utf-8;'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${datasetId}_${m.id}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('CSV exported with active problem and model.');
};
setupDataset();
})();
