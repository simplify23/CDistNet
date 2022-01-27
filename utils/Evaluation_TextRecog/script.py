#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from StringIO import StringIO
import rrc_evaluation_funcs
import importlib
import pdb
def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation. 
    """    
    return {
            'xlsxwriter':'xlsxwriter',
            'editdistance':'editdistance'
            }

def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """        
    return {
            'SAMPLE_NAME_2_ID':'(?:word_)?([0-9]+).png',
            'CRLF':False,
            'DOUBLE_QUOTES':True
            }

def validate_data(gtFilePath, submFilePath,evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    
    gtFile = rrc_evaluation_funcs.decode_utf8(open(gtFilePath,'rb').read())
    if (gtFile is None) :
        raise Exception("The GT file is not UTF-8")
        
    gtLines = gtFile.split( "\r\n" if evaluationParams['CRLF'] else "\n" )
    ids = {}
    for line in gtLines:
        line = line.replace("\r","").replace("\n","")
        if(line != ""):
            if (evaluationParams['DOUBLE_QUOTES']):
                m = re.match(r'^' + evaluationParams['SAMPLE_NAME_2_ID'] + ',\s?\"(.*)\"\s*\t?$',line)
            else:
                m = re.match(r'^' + evaluationParams['SAMPLE_NAME_2_ID'] + ',\s?(.*)$',line)
                
            if m == None :
                if (evaluationParams['DOUBLE_QUOTES']):
                    raise Exception(("Line in GT not valid.Found: %s should be: %s"%(line,evaluationParams['SAMPLE_NAME_2_ID'] + ',transcription' )).encode('utf-8', 'replace'))            
                else:
                    raise Exception(("Line in GT not valid.Found: %s should be: %s"%(line,evaluationParams['SAMPLE_NAME_2_ID'] + ',"transcription"' )).encode('utf-8', 'replace'))            
            ids[m.group(1)] = {'gt':m.group(2),'det':''}
    
    submFile = rrc_evaluation_funcs.decode_utf8(open(submFilePath,'rb').read())
    if (submFile is None) :
        raise Exception("The Det file is not UTF-8")
    
    submLines = submFile.split("\r\n" if evaluationParams['CRLF'] else "\n")
    for line in submLines:
        line = line.replace("\r","").replace("\n","")
        if(line != ""):
            if (evaluationParams['DOUBLE_QUOTES']):
                m = re.match(r'^' + evaluationParams['SAMPLE_NAME_2_ID'] + ',\s?\"(.*)\"\s*\t?$',line)
            else:
                m = re.match(r'^' + evaluationParams['SAMPLE_NAME_2_ID'] + ',\s?(.*)$',line)

            if m == None :
                if (evaluationParams['DOUBLE_QUOTES']):
                    raise Exception(("Line in results not valid.Found: %s should be: %s"%(line,evaluationParams['SAMPLE_NAME_2_ID'] + ',transcription' )).encode('utf-8', 'replace'))            
                else:
                    raise Exception(("Line in results not valid.Found: %s should be: %s"%(line,evaluationParams['SAMPLE_NAME_2_ID'] + ',"transcription"' )).encode('utf-8', 'replace'))  
            try:
                ids[m.group(1)]['det'] = m.group(2)
            except Exception as e:
                raise Exception(("Line in results not valid. Line: %s Sample item not valid: %s" %(line,m.group(1))).encode('utf-8', 'replace'))

def evaluate_method(gtFilePath, submFilePath,evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """  
    
    for module,alias in evaluation_imports().iteritems():
        globals()[alias] = importlib.import_module(module) 
    
    gtFile = rrc_evaluation_funcs.decode_utf8(open(gtFilePath,'rb').read())
    gtLines = gtFile.split("\r\n" if evaluationParams['CRLF'] else "\n")#'CRLF':False,
    ids = {}
    for line in gtLines:
        line = line.replace("\r","").replace("\n","")
        if(line != ""):
            if (evaluationParams['DOUBLE_QUOTES']):#'DOUBLE_QUOTES':True
                m = re.match(r'^' + evaluationParams['SAMPLE_NAME_2_ID'] + ',\s?\"(.+)\"$',line)
                ids[m.group(1)] = {"gt" : m.group(2).replace("\\\\", "\\").replace("\\\"", "\""),"det":""}
            else:
                m = re.match(r'^' + evaluationParams['SAMPLE_NAME_2_ID'] + ',\s?(.+)$',line)
                ids[m.group(1)] = {"gt" :m.group(2),"det":""}
    
            
    totalDistance = 0.0
    totalLength = 0.0
    totalDistanceUpper = 0.0
    totalLengthUpper = 0.0
    numWords = 0
    correctWords = 0.0
    correctWordsUpper = 0.0
    
    perSampleMetrics = {}    
    
    submFile = rrc_evaluation_funcs.decode_utf8(open(submFilePath,'rb').read())
    if (submFile is None) :
        raise Exception("The file is not UTF-8")
    
    xls_output = StringIO()    
    workbook = xlsxwriter.Workbook(xls_output)
    worksheet = workbook.add_worksheet()    
    worksheet.write(1, 1 , "sample")
    worksheet.write(1, 2 , "gt")
    worksheet.write(1, 3 , "E.D.")
    worksheet.write(1, 4 , "normalized")
    worksheet.write(1, 5 , "E.D. upper")
    worksheet.write(1, 6 , "normalized upper")
    
    submLines = submFile.split("\r\n" if evaluationParams['CRLF'] else "\n")
    for line in submLines:
        line = line.replace("\r","").replace("\n","")
        if(line != ""):
            
            numWords = numWords + 1
            
            if (evaluationParams['DOUBLE_QUOTES']):
                m = re.match(r'^' + evaluationParams['SAMPLE_NAME_2_ID'] + ',\s?\"(.*)\"\s*\t?$',line)
                detected = m.group(2).replace("\\\\", "\\").replace("\\\"", "\"")
            else:
                m = re.match(r'^' + evaluationParams['SAMPLE_NAME_2_ID'] + ',\s?(.*)$',line)
                detected = m.group(2)
            
            ids[m.group(1)]['det'] = detected
        
    row = 1
    for k,v in ids.iteritems():
        
        gt = v['gt'] 
        detected = v['det'] 

        if gt == detected : 
            correctWords = correctWords + 1

        if gt.upper() == detected.upper() : 
            correctWordsUpper = correctWordsUpper + 1                

        '''
        distance = editdistance.eval(gt, detected)
        length = float(distance) / len (gt )

        distance_up = editdistance.eval(gt.upper(), detected.upper())
        length_up = float(distance_up) / len (gt )

        totalDistance += distance
        totalLength += length

        totalDistanceUpper += distance_up
        totalLengthUpper += length_up
        '''

        distance = editdistance.eval(gt, detected)
        length = len (gt )

        distance_up = editdistance.eval(gt.upper(), detected.upper())
        length_up = len (gt )

        totalDistance += distance
        totalLength += length

        totalDistanceUpper += distance_up
        totalLengthUpper += length_up


        perSampleMetrics[k] = {
                                'gt':gt,
                                'det':detected,
                                'edist':distance,
                                'norm':length ,
                                'edistUp':distance_up,
                                'normUp':length_up 
                                }
        row = row + 1
        worksheet.write(row, 1, k)
        worksheet.write(row, 2, gt)
        worksheet.write(row, 3, detected)
        worksheet.write(row, 4, distance)
        worksheet.write(row, 5, length)
        worksheet.write(row, 6, distance_up)
        worksheet.write(row, 7, length_up)
                                
    methodMetrics = {
                    'totalWords':len(ids),
                    'detWords':numWords,
                    'crwN':correctWords,
                    'crwupN':correctWordsUpper,
                    'ted':totalDistance,
                    'tedL': 1.0 -(float(totalDistance) / totalLength),
                    'crw':0 if numWords==0 else correctWords/numWords,
                    'crwN':correctWords,
                    'tedup':totalDistanceUpper,
                    'tedupL': 1.0 - (float(totalDistanceUpper) / totalLengthUpper),
                    'crwup':0 if numWords==0 else correctWordsUpper/numWords,
                    'crwupN':correctWordsUpper
                    }

    workbook.close()
    output_items = {'samples.xlsx':xls_output.getvalue()}
    xls_output.close()                                

    resDict = {'calculated':True,'Message':'','method': methodMetrics,'per_sample': perSampleMetrics,'output_items':output_items}
    return resDict;

if __name__=='__main__':
    rrc_evaluation_funcs.main_evaluation(None,default_evaluation_params,validate_data,evaluate_method)
