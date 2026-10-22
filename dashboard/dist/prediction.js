/* Verified static inference representation exported from fitted sklearn/OLS. */
(function(root){
'use strict';
const missing=v=>v===null||v===undefined||v==='';
const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
function encode(descriptor,raw){
 const result={};let offset=0;
 descriptor.numericColumns.forEach((name,i)=>{
  const value=missing(raw[name])?descriptor.numericMedian[i]:Number(raw[name]);
  if(!Number.isFinite(value))throw new Error('Non-finite numeric input: '+name);
  result[descriptor.featureNames[offset++]]=(value-descriptor.numericMean[i])/descriptor.numericScale[i];
 });
 descriptor.categoricalColumns.forEach((name,i)=>{
  let value=raw[name];
  if(name===descriptor.manufacturerColumn&&!missing(value)){
   value=String(value).trim().toLowerCase().split(/\s+/)[0];value=descriptor.manufacturerAliases[value]||value;
  }
  if(missing(value))value=descriptor.categoricalFill[i];
  descriptor.categories[i].forEach(category=>{result[descriptor.featureNames[offset++]]=+(String(value)===String(category));});
 });
 return result;
}
function predict(dataset,model,raw){
 const encoded=encode(dataset.preprocessing,raw);
 return model.intercept+model.weights.reduce((sum,w,i)=>sum+w*encoded[model.features[i]],0);
}
function quadratic(vector,matrix){return dot(vector,matrix.map(row=>dot(row,vector)));}
function intervals(dataset,model,raw){
 const encoded=encode(dataset.preprocessing,raw),z=model.features.map(n=>encoded[n]);
 const u=model.uncertainty,prediction=model.intercept+dot(model.weights,z),difference=z.map((v,i)=>v-u.center[i]);
 const scores=u.vectors.map(row=>dot(row,difference));
 const projected=difference.map((_,j)=>scores.reduce((sum,v,i)=>sum+v*u.vectors[i][j],0));
 const error=Math.hypot(...difference.map((v,i)=>v-projected[i]));
 const estimable=error<=u.estimabilityTolerance*(1+Math.hypot(...difference));
 if(!estimable)return {prediction,estimable:false,status:'outside_training_affine_span'};
 const basis=[1,...scores.map((v,i)=>v/u.scales[i])];
 const variance=Math.max(0,quadratic(basis,u.covariance)),halfMean=u.tCritical*Math.sqrt(variance),halfPI=u.tCritical*Math.sqrt(variance+u.sigma2);
 const robust=u.robustCovariance?u.zCritical*Math.sqrt(Math.max(0,quadratic(basis,u.robustCovariance))):null;
 return {prediction,estimable:true,meanLower:prediction-halfMean,meanUpper:prediction+halfMean,
  piLower:prediction-halfPI,piUpper:prediction+halfPI,
  robustMeanLower:robust===null?null:prediction-robust,robustMeanUpper:robust===null?null:prediction+robust,
  leverage:variance/u.sigma2,robustType:u.robustType};
}
function validate(dataset,raw){
 const errors=[],warnings=[];
 dataset.inputs.forEach(f=>{
  const value=raw[f.name];
  if(f.kind==='numeric'){
   if(missing(value)||!Number.isFinite(Number(value))){errors.push(f.label+': enter a finite number.');return;}
   const number=Number(value);
   if(Math.abs(number)>1e9||(f.integer&&!Number.isInteger(number))||(f.domainMin!==null&&number<f.domainMin)||(f.domainMax!==null&&number>f.domainMax))errors.push(f.label+': outside the valid domain'+(f.integer?' or not an integer.':'.'));
   else if(number<f.min||number>f.max)warnings.push(f.label+': outside observed training range ['+f.min+', '+f.max+'].');
  }else if(missing(value))errors.push(f.label+': select a category.');
  else if(!f.options.includes(String(value)))warnings.push(f.label+': unknown category; encoded with zero indicators.');
 });
 return {errors,warnings};
}
root.DashboardPrediction={encode,predict,intervals,validate};
})(globalThis);
