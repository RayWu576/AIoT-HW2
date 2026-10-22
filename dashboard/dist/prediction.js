/* The browser executes the training-fitted sklearn preprocessing descriptor. */
(function(root) {
'use strict';
const missing = v => v === null || v === undefined || v === '';
function encode(descriptor, raw) {
 const result = {};
 let offset = 0;
 descriptor.numericColumns.forEach((name,i) => {
  const value = missing(raw[name]) ? descriptor.numericMedian[i] : Number(raw[name]);
  result[descriptor.featureNames[offset++]] = (value-descriptor.numericMean[i])/descriptor.numericScale[i];
 });
 descriptor.categoricalColumns.forEach((name,i) => {
  let value = raw[name];
  if(name === descriptor.manufacturerColumn && !missing(value)) {
   value = String(value).trim().toLowerCase().split(/\s+/)[0];
   value = descriptor.manufacturerAliases[value] || value;
  }
  if(missing(value)) value = descriptor.categoricalFill[i];
  descriptor.categories[i].forEach(category => {
   result[descriptor.featureNames[offset++]] = +(String(value) === String(category));
  });
 });
 return result;
}
function predict(dataset, model, raw) {
 const encoded = encode(dataset.preprocessing,raw);
 return model.intercept + model.weights.reduce((sum,w,i) => sum+w*encoded[model.features[i]],0);
}
root.DashboardPrediction = {encode,predict};
})(globalThis);
