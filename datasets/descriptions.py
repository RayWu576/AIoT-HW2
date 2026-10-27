"""Presentation definitions and physical input constraints, not fitted statistics."""
STUDENT_LABELS = {
 'school':'School', 'sex':'Sex', 'age':'Age', 'address':'Home address', 'famsize':'Family size',
 'Pstatus':'Parent cohabitation', 'Medu':'Mother education (0-4)', 'Fedu':'Father education (0-4)',
 'Mjob':'Mother job', 'Fjob':'Father job', 'reason':'School choice reason', 'guardian':'Guardian',
 'traveltime':'Travel time category (1-4)', 'studytime':'Study time category (1-4)',
 'failures':'Past failures category', 'schoolsup':'School educational support', 'famsup':'Family educational support',
 'paid':'Paid classes', 'activities':'Extracurricular activities', 'nursery':'Attended nursery',
 'higher':'Higher education intention', 'internet':'Internet access', 'romantic':'Romantic relationship',
 'famrel':'Family relationships (1-5)', 'freetime':'Free time (1-5)', 'goout':'Going out (1-5)',
 'Dalc':'Workday alcohol (1-5)', 'Walc':'Weekend alcohol (1-5)', 'health':'Health (1-5)', 'absences':'Absences',
}
ROOMS={1:'Kitchen',2:'Living room',3:'Laundry',4:'Office',5:'Bathroom',6:'Outside north side',7:'Ironing room',8:'Teenager room',9:'Parents room'}

CAR_FIELDS = {
 'symboling':('Insurance risk rating','code'), 'CarName':('Manufacturer (from CarName)',''),
 'fueltype':('Fuel type',''), 'aspiration':('Aspiration',''), 'doornumber':('Number of doors',''),
 'carbody':('Body style',''), 'drivewheel':('Drive wheels',''), 'enginelocation':('Engine location',''),
 'wheelbase':('Wheelbase','in'), 'carlength':('Car length','in'), 'carwidth':('Car width','in'),
 'carheight':('Car height','in'), 'curbweight':('Curb weight','lb'), 'enginetype':('Engine type',''),
 'cylindernumber':('Number of cylinders',''), 'enginesize':('Engine size','cu in'),
 'fuelsystem':('Fuel system',''), 'boreratio':('Bore diameter','in'), 'stroke':('Stroke','in'),
 'compressionratio':('Compression ratio','ratio'), 'horsepower':('Horsepower','hp'),
 'peakrpm':('Peak engine speed','rpm'), 'citympg':('City fuel economy','mpg'),
 'highwaympg':('Highway fuel economy','mpg'),
}

def field_definition(config,name):
    if config.dataset_id=='car_price':
        label,unit=CAR_FIELDS[name]
        numeric=name in config.numeric_columns
        low,high=(-3,3) if name=='symboling' else (0,None) if numeric else (None,None)
        return {'label':label,'unit':unit,'domainMin':low,'domainMax':high,'integer':name=='symboling',
                'definition':label+'; original vehicle specification.'}

    if config.dataset_id=='student_performance':
        bounds={'age':(1,100),'Medu':(0,4),'Fedu':(0,4),'traveltime':(1,4),'studytime':(1,4),
                'failures':(0,4),'absences':(0,10000)}
        for column in ('famrel','freetime','goout','Dalc','Walc','health'):bounds[column]=(1,5)
        low,high=bounds.get(name,(None,None))
        return {'label':STUDENT_LABELS.get(name,name),'unit':'years' if name=='age' else 'days' if name=='absences' else '',
                'domainMin':low,'domainMax':high,'integer':name in config.numeric_columns,
                'definition':STUDENT_LABELS.get(name,name)+'; original questionnaire/academic background field.'}
    if name.startswith('RH_') and name[3:].isdigit():
        return {'label':ROOMS[int(name[3:])]+' humidity','unit':'%','domainMin':0,'domainMax':100,'integer':False,'definition':'Relative humidity sensor, '+ROOMS[int(name[3:])]+'.'}
    if name.startswith('T') and name[1:].isdigit():
        return {'label':ROOMS[int(name[1:])]+' temperature','unit':'C','domainMin':-273.15,'domainMax':100,'integer':False,'definition':'Temperature sensor, '+ROOMS[int(name[1:])]+'.'}
    extra={'lights':('Lighting energy','Wh',0,None),'T_out':('Weather station temperature','C',-273.15,100),
           'Press_mm_hg':('Weather station pressure','mm Hg',1,None),'RH_out':('Outdoor humidity','%',0,100),
           'Windspeed':('Wind speed','m/s',0,None),'Visibility':('Visibility','km',0,None),
           'Tdewpoint':('Dew point','C',-273.15,100)}
    label,unit,low,high=extra[name]
    return {'label':label,'unit':unit,'domainMin':low,'domainMax':high,'integer':False,'definition':label+'; contemporaneous observation, not a future measurement.'}

# Presentation metadata shared by official exports and documentation.
DATASET_PRESENTATION = {
 'student_performance': {
  'title':'Student Performance Prediction','problem':'Problem 1: Student Performance Prediction',
  'description':'Early prediction of Mathematics final grade from background and study characteristics.',
  'sampleLabel':'Student','source':'https://archive.ics.uci.edu/dataset/320/student+performance',
  'kaggleSource':'https://www.kaggle.com/datasets/larsen0966/student-performance-data-set',
  'sourceNote':'Requested Kaggle archive supplies Portuguese data only; authentic Mathematics file comes from UCI (395 rows).',
  'targetNote':'G1/G2 are historical grades and excluded from the official early-30 scenario.'},
 'appliances_energy': {
  'title':'Appliances Energy Prediction','problem':'Problem 2: Appliances Energy Prediction',
  'description':'Predict appliance energy use from contemporaneous household sensors and weather.',
  'sampleLabel':'Observation','source':'https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction',
  'kaggleSource':'https://www.kaggle.com/datasets/loveall/appliances-energy-prediction',
  'sourceNote':'Authentic Kaggle copy; date retained only as metadata, rv1/rv2 excluded; lights is retained.',
  'targetNote':'Chronological residuals are dependent; intervals are working-model approximations, not forecast guarantees.'},
 'car_price': {
  'title':'Car Price Prediction','problem':'Problem 3: Car Price Prediction',
  'description':'Estimate vehicle price from specifications and a manufacturer derived from CarName.',
  'sampleLabel':'Vehicle','source':'https://www.kaggle.com/datasets/hellbuoy/car-price-prediction',
  'kaggleSource':'https://www.kaggle.com/datasets/hellbuoy/car-price-prediction',
  'sourceNote':'Authentic Kaggle CarPrice_Assignment.csv: 205 rows, 26 columns. Exclude car_ID and price; retain 24 original candidate inputs.',
  'targetNote':'Price is in USD. Only 205 observations; manufacturer replaces the high-cardinality vehicle name inside each training-fold Pipeline. This is not a current-market valuation guarantee.'},
}
