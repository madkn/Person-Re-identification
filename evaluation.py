import numpy as np
import os

from scipy.spatial.distance import cdist

def make_index(test_image_names, query_image_names):
    Index = dict()
    Index['junk'] = set()
    Index['distractor'] = set()
    for name in test_image_names:
        if ifJunk(name):
            Index['junk'].add(name)
        elif ifDistractor(name):
            Index['distractor'].add(name)
    for query in query_image_names:
        Index[query] = dict()
        Index[query]['pos'] = set()
        Index[query]['junk'] = set()

        person, camera = parse_market_1501_name(query)
        for name in test_image_names:
            if ifJunk(name) or ifDistractor(name):
                continue
            person_, camera_ = parse_market_1501_name(name)
            if person == person_ and camera != camera_ :
                Index[query]['pos'].add(name)

            elif person == person_ and camera == camera :
                Index[query]['junk'].add(name)
                
    return Index

def parse_market_1501_name(full_name):
    name_ar = full_name.split('/')
    name = name_ar[len(name_ar)-1]
    
    person = int(name.split('_')[0])
    camera = int(name.split('_')[1].split('s')[0].split('c')[1])
    
    return person, camera

def parseMarket1501(path):
    person_label = []
    camera_label = []
    image_path = []
    image_name = []

    for file in sorted(os.listdir(path)):
        if file.endswith(".jpg"):
            person, camera = parse_market_1501_name(file)
            
            person_label.append(person)
            camera_label.append(camera)
            image_path.append(os.path.join(path, file))
            image_name.append(file)
            
    return person_label, camera_label, image_path, image_name

def ifJunk(filename):
    if filename.startswith("-1"):
        return True
    else:
        return False
    
def ifDistractor(filename):
    if filename.startswith("0000"):
        return True
    else:
        return False
    
def getPlace(query, sorted_gallery_filenames, Index):    
        
    place = 0
    for i in range(len(sorted_gallery_filenames)):
        if sorted_gallery_filenames[i] in Index['junk'] or sorted_gallery_filenames[i] in Index[query]['junk']:
            continue
        elif     sorted_gallery_filenames[i] in Index['distractor']:
            place +=1
           
        elif sorted_gallery_filenames[i] in Index[query]['pos']:
           # print "PLACE " , sorted_gallery_filenames[i]
            return place
        else :
            place +=1
            
    return place  

def ranking(metric, gescrs_query, query_image_names, gescrs_gallery, test_image_names, maxrank, Index):
    ranks = np.zeros(maxrank + 1)
    places = dict()
    print('Calculating distances')
    all_dist = cdist(gescrs_query, gescrs_gallery, metric)
    np_test_image_names = np.array(test_image_names)
    img_names_sorted = dict()
    
    all_gallery_names_sorted = np_test_image_names[np.argsort(all_dist).astype(np.uint32)]
    for qind in range(len(gescrs_query)):       
        dist = all_dist[qind]
        gallery_names_sorted = all_gallery_names_sorted[qind]
      
        place=getPlace(query_image_names[qind], gallery_names_sorted, Index)       
        img_names_sorted[qind] = all_gallery_names_sorted[qind]
                
        places[qind] = place

        ranks[place+1:maxrank+1] += 1
        
    return ranks, img_names_sorted,places

def cos_dist(x, y):
    xy = np.dot(x,y);
    xx = np.dot(x,x);
    yy = np.dot(y,y);  
            
    return -xy*1.0/np.sqrt(xx*yy)

def getDistances(gescr_query, gescrs_gallery):
    dist = list()
    
    for i in range(len(gescrs_gallery)):
        dist.append(cos_dist(gescr_query, gescrs_gallery[i]))
        
    return dist

def getAveragePrecision(query, sorted_gallery_filenames, Index):    
        
    ap = 0
    tp = 0
    k = 0
    
    for i in range(len(sorted_gallery_filenames)):
        
        if sorted_gallery_filenames[i] in Index['junk'] or sorted_gallery_filenames[i] in Index[query]['junk']:
            continue
        elif     sorted_gallery_filenames[i] in Index['distractor']:
            k+=1
            deltaR = 0
        elif sorted_gallery_filenames[i] in Index[query]['pos']:
            tp+=1
            k+=1
            deltaR = 1.0/len(Index[query]['pos'])
        else :
            k +=1
            deltaR = 0
        
        if tp == len(Index[query]['pos']):
            return ap
        precision = tp*1.0/k * deltaR
        ap += precision
        
    return ap

def mAP(gescrs_query, query_image_names, gescrs_gallery, test_image_names, maxrank, Index):
    ranks = np.zeros(maxrank+1)
    places = dict()
    ap_sum = 0
    
    for qind in tqdm(range(len(gescrs_query))):       
        dist = getDistances(gescrs_query[qind], gescrs_gallery)
        dist_zip = sorted(zip(dist,test_image_names))
        gallery_names_sorted = [x for (y,x) in dist_zip]
      
        ap=getAveragePrecision(query_image_names[qind], gallery_names_sorted, Index)
        ap_sum += ap
        
    return ap_sum * 1.0 /len(gescrs_query)

def parse_market_1501_name(full_name):
    name_ar = full_name.split('/')
    name = name_ar[len(name_ar)-1]
    
    person = int(name.split('_')[0])
    camera = int(name.split('_')[1].split('s')[0].split('c')[1])
    
    return person, camera

def parseMarket1501(path):
    person_label = []
    camera_label = []
    image_path = []
    image_name = []

    for file in sorted(os.listdir(path)):
        if file.endswith(".jpg"):
            person, camera = parse_market_1501_name(file)
            
            person_label.append(person)
            camera_label.append(camera)
            image_path.append(os.path.join(path, file))
            image_name.append(file)
            
    return person_label, camera_label, image_path, image_name

def ifJunk(filename):
    if filename.startswith("-1"):
        return True
    else:
        return False
    
def ifDistractor(filename):
    if filename.startswith("0000"):
        return True
    else:
        return False
    
def getPlace(query, sorted_gallery_filenames, Index):    
        
    place = 0
    for i in range(len(sorted_gallery_filenames)):
        if sorted_gallery_filenames[i] in Index['junk'] or sorted_gallery_filenames[i] in Index[query]['junk']:
            continue
        elif     sorted_gallery_filenames[i] in Index['distractor']:
            place +=1
           
        elif sorted_gallery_filenames[i] in Index[query]['pos']:
           # print "PLACE " , sorted_gallery_filenames[i]
            return place
        else :
            place +=1
            
    return place  

def ranking(metric, gescrs_query, query_image_names, gescrs_gallery, test_image_names, maxrank, Index):
    ranks = np.zeros(maxrank + 1)
    places = dict()
    #print('Calculating distances')
    all_dist = cdist(gescrs_query, gescrs_gallery, metric)
    np_test_image_names = np.array(test_image_names)
    img_names_sorted = dict()
    
    all_gallery_names_sorted = np_test_image_names[np.argsort(all_dist).astype(np.uint32)]
    for qind in range(len(gescrs_query)):       
        dist = all_dist[qind]
        gallery_names_sorted = all_gallery_names_sorted[qind]
      
        place=getPlace(query_image_names[qind], gallery_names_sorted, Index)       
        img_names_sorted[qind] = all_gallery_names_sorted[qind]
                
        places[qind] = place

        ranks[place+1:maxrank+1] += 1
        
    return ranks, img_names_sorted,places

def cos_dist(x, y):
    xy = np.dot(x,y);
    xx = np.dot(x,x);
    yy = np.dot(y,y);  
            
    return -xy*1.0/np.sqrt(xx*yy)

def getDistances(gescr_query, gescrs_gallery):
    dist = list()
    
    for i in range(len(gescrs_gallery)):
        dist.append(cos_dist(gescr_query, gescrs_gallery[i]))
        
    return dist

def getAveragePrecision(query, sorted_gallery_filenames, Index):    
        
    ap = 0
    tp = 0
    k = 0
    
    for i in range(len(sorted_gallery_filenames)):
        
        if sorted_gallery_filenames[i] in Index['junk'] or sorted_gallery_filenames[i] in Index[query]['junk']:
            continue
        elif     sorted_gallery_filenames[i] in Index['distractor']:
            k+=1
            deltaR = 0
        elif sorted_gallery_filenames[i] in Index[query]['pos']:
            tp+=1
            k+=1
            deltaR = 1.0/len(Index[query]['pos'])
        else :
            k +=1
            deltaR = 0
        
        if tp == len(Index[query]['pos']):
            return ap
        precision = tp*1.0/k * deltaR
        ap += precision
        
    return ap

def mAP(gescrs_query, query_image_names, gescrs_gallery, test_image_names, maxrank, Index):
    ranks = np.zeros(maxrank+1)
    places = dict()
    ap_sum = 0
    
    for qind in range(len(gescrs_query)):       
        dist = getDistances(gescrs_query[qind], gescrs_gallery)
        dist_zip = sorted(zip(dist,test_image_names))
        gallery_names_sorted = [x for (y,x) in dist_zip]
      
        ap=getAveragePrecision(query_image_names[qind], gallery_names_sorted, Index)
        ap_sum += ap
        
    return ap_sum * 1.0 /len(gescrs_query)
