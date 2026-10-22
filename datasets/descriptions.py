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

def field_definition(config,name):
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
