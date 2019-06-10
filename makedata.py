import pandas as pd
from pathlib import Path
import numpy as np
import math
import pickle 
path = Path('./')
train_path = path.joinpath('train.csv')
test_path = path.joinpath('test.csv')
submission_path = path.joinpath('submission_popular.csv')
metadata_path = path.joinpath('item_metadata.csv')

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_meta = pd.read_csv(metadata_path)
df_submission = pd.read_csv(submission_path)

action_list = set(df_train['action_type'].values)
reference_list = set(df_train['reference'].values)
platform_list = set(df_train['platform'].values)
city_list = set(df_train['city'].values)
device_list = set(df_train['device'].values)
filter_list = set(df_train['current_filters'].values)
filter_list =  [x for x in filter_list if str(x) != 'nan']

action_list = list(action_list)
action_list.sort()

reference_list = list(reference_list)
reference_list.sort()

platform_list = list(platform_list)
platform_list.sort()

city_list = list(city_list)
city_list.sort()

device_list = list(device_list)
device_list.sort()

filter_list = list(filter_list)
filter_list.sort()

#테스트 데이터에서 마지막 reference가 nan인지 확인
def check_data(data):
    max_error = 0
    datalength = len(data)
    for i in range(datalength):
        if(isinstance(data[i][-1,5], float)):
            if(math.isnan(data[i][-1,5]) == True):
                pass
            else:
                print('error')
                print(data[i])
                max_error +=1
        else:
            print('error')
            print(data[i])
            max_error +=1
        if(max_error > 10):
            break
            
#테스트 데이터에서 reference가 nan일때 그 전 action이 click out인지 확인
def check_data_alpha(data):
    max_error = 0
    datalength = len(data)
    for i in range(datalength):
        for j in range(len(data[i])):
            if(isinstance(data[i][j,5], float)):
                if(math.isnan(data[i][j,5]) == True):
                     if(data[i][j,4] != 'clickout item'):
                        print(data[i])
                        max_error +=1
        
        if(max_error > 10):
            break
            
            
#테스트 데이터에서 reference에 하나라도 nan이 있는지 판별 있다면 그 중 적어도 하나는 action이 click out인지 판별
def check_data_beta(data):
    max_error = 0
    datalength = len(data)
    for i in range(datalength):
        error_data_number = 0
        click_check = False
        for j in range(len(data[i])):
            if(isinstance(data[i][j,5], float) and math.isnan(data[i][j,5]) == True):
                error_data_number += 1
        if(error_data_number == 0):
            print(data[i])
            print('error')
            max_error +=1
        if(error_data_number != 0):
            for j in range(len(data[i])):
                if(isinstance(data[i][j,5], float) and math.isnan(data[i][j,5]) == True):
                    if(data[i][j,4] == 'clickout item'):
                        click_check = True
        if(click_check == False):
            print('error')
            print(data[i])
            max_error +=1
            #print(data[i])
            #print('datas')
        if(max_error > 10):
            break
            
            
            
            
            
# 트레이닝 데이터를 세션 단위로 나눔
def make_data(data): 
    data_split = []
    temp_data = []
    train_size = len(data)
    action_index = 1
    for i in range(train_size):
        if(i==0):
            temp_data.append(data[i])
        if(i>0):
            if(data[i,action_index] == data[i-1,action_index]):
                temp_data.append(data[i])
            else:
                data_split.append(np.array(temp_data))
                temp_data = []
                temp_data.append(data[i])
    return data_split

train_data_split = make_data(df_train.values)
#트레이닝 데이터에서 click out이 존재하는 데이터만 뽑아냄
#트레이닝 데이터에서 끝부분이 click out이 되도록 뒷부분을 자름
def data_preprocess(data): 
    new_data = []
    for i in range(len(data)):
        check_data = False
        final_click = 0
        for j in range(len(data[i])):
            if(data[i][j,4] =='clickout item'):
                final_click = j
                check_data = True
        if(final_click < len(data[i])-1 and check_data):
            new_data.append(np.delete(data[i],np.s_[final_click+1:],0))
        elif(check_data):
            new_data.append(data[i])
    return new_data

train_data_preprocess = data_preprocess(train_data_split)
del train_data_split
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
#reference embedding 보조 함수
items = df_meta['item_id'].values
properties = df_meta['properties'].values
all_items_test = []
for i in range(len(properties)):
    p_data = properties[i].split('|')
    all_items_test += p_data
set_items = set(all_items_test)

list_items = list(set_items)
list_items.sort()
def embedding_item_diction():
    all_items = {}
    for i in range(len(properties)):
        all_items[items[i]] = properties[i].split('|')
    embedding_items = {}
    for keys in all_items.keys():
        item_embeddings = [0]*len(list_items)
        for i in range(len(list_items)):
            if list_items[i] in all_items[keys]:
                item_embeddings[i] = 1
        embedding_items[keys] = item_embeddings
    return embedding_items
reference_diction = embedding_item_diction()



#filter embedding 보조함수
all_filters = []
for i in range(len(filter_list)):
    if i>0:
        all_filters += filter_list[i].split('|')
all_filter_set = set(all_filters)
all_filter_list = list(all_filter_set)
all_filter_list.sort()
def filter_embedding(data, version):
    embedding_data = [0]*202
    if version == 1:
        datalist = data.split('|')
        for filters in datalist:
            embedding_data[all_filter_list.index(filters)] = 1
        return embedding_data
    
    
    
def one_hot_encoding(number, length):
    return [int(i==number) for i in range(length)]




# 전체 action embedding
def embedding(data, action_index):
    if action_index == 0: #user_id dim 0
        return []
    
    if action_index == 1: # session_id dim 0
        return []
    
    if action_index == 2: # timestep dim 0
        return []
    
    if action_index == 3: #step dim 1
        return []
    
    if action_index == 4: #action_type dim 10
        #print(one_hot_encoding(action_list.index(data), 10))
        return one_hot_encoding(action_list.index(data), 10)
    
    if action_index == 5: #reference dim 157
        if RepresentsInt(data):
            return reference_diction[int(data)]
        else:
            return [0]*157
        
    if action_index == 6: #platform dim 55
        return one_hot_encoding(platform_list.index(data), 55)
    
    if action_index == 7: #city dim 0
        return []
    
    if action_index == 8: #device dim 3
        return one_hot_encoding(device_list.index(data), 3)
    
    if action_index == 9: #current_filters dim 202
        if(isinstance(data, float) and math.isnan(data) == True):
            return [0]*202
        else:
            return filter_embedding(data, 1)
        
    if action_index == 10: #impressions dim 3925
        if(isinstance(data, float) and math.isnan(data) == True):
            return [0]*3925
        else:
            impression_embedding = []
            impression_list = data.split('|')
            for impression in impression_list:
                try:
                    impression_embedding += reference_diction[int(impression)]
                except:
                    pass
            if(len(impression_embedding) < 3925):
                impression_embedding += [0]*(3925-len(impression_embedding))   
            return impression_embedding
        
    if action_index == 11: #prices dim 25
        if(isinstance(data, float) and math.isnan(data) == True):
            return [0]*25
        else:
            price_embedding = []
            for prices in data.split('|'):
                price_embedding.append(int(prices))
            if(len(price_embedding) < 25):
                price_embedding += [0]*(25-len(price_embedding))   
            return price_embedding
        
        
        
def check_last_click(data):
    data_number = 0
    for i in range(len(data)):
        if(data[i][-1,4] == 'clickout item'):
            data_number +=1
    return data_number




action_type_length = 10
reference_length = 157
platform_length = 55
device_length = 3
current_filters_length = 202
impresssions_length = 3925
prices_length = 25
def addVector(vector1, vector2, coef, action_num):
    new_action_type = []
    new_reference = []
    new_platform = []
    new_device = []
    new_filters = []
    new_impressions = []
    new_prices = [] 
    #print(len(vector1))
    #print(len(vector2))
    for i in range(10):
        new_action_type.append(vector1[i] + np.power(coef,action_num)*vector2[i])
    
    for i in range(10, 167):
        new_action_type.append(vector1[i] + np.power(coef,action_num)*vector2[i])
        
    for i in range(167, 222):
        new_action_type.append(vector1[i])
    
    for i in range(222, 225):
        new_action_type.append(vector1[i])
        
    for i in range(225, 427):
        new_action_type.append(vector1[i] + np.power(coef,action_num)*vector2[i])
    
    for i in range(427, 4352):
        new_action_type.append(vector1[i] + np.power(coef,action_num)*vector2[i])
        
    for i in range(4352, 4377):
        new_action_type.append(vector1[i] + np.power(coef,action_num)*vector2[i])
        
    return new_action_type




def MakeEmbeddingVector(data):
    #embedding_vector = []
    #temp_vector = []
    session_vector = [0]*4377 # session data embedding(length 10+157+55+3+202+3925+25=4377)
    item_vector = [] # item data embedding
    num_action = len(data)
    time_delay_coef = 0.1
    for i in range(num_action-1, -1, -1): #거꾸로
        timestep_vector = []
        for j in range(12):
            timestep_vector += embedding(data[i,j], j) 
        if(i == num_action-1):
            for idx in range(10, 167):
                timestep_vector[idx] = 0 # reference 부분을 0으로 만듦
            item_vector = embedding(data[i, 5], 5) #item vector 생성
        session_vector = addVector(session_vector, timestep_vector, time_delay_coef, num_action-1-i) #session vector 생성   
            
    return session_vector, item_vector



final_train_data = []
final_train_label = []
datalength = len(train_data_preprocess)

del df_train
del df_test
del df_meta
del df_submission
# 메모리 관리중

num_error = 0
for i in range(datalength//2):
    if(i%10000 ==0):
        print('making data....', i)
    try:
        temp_train, temp_label = MakeEmbeddingVector(train_data_preprocess[i])
    except:
        num_error += 1
    final_train_data.append(temp_train)
    final_train_label.append(temp_label)
    
final_train_data = np.asarray(final_train_data)
final_train_label = np.asarray(final_train_label)

with open('final_train_data1.pickle', 'wb') as f:
    pickle.dump(final_train_data, f, pickle.HIGHEST_PROTOCOL)
    
with open('final_train_label1.pickle', 'wb') as f:
    pickle.dump(final_train_label, f, pickle.HIGHEST_PROTOCOL)    

print('number of error is : ', num_error)
print('total data is : ', len(final_train_data))    
    
final_train_data = []
final_train_label = []

for i in range(datalength//2,len(train_data_preprocess)):
    if(i%10000 ==0):
        print('making data....', i)
    try:
        temp_train, temp_label = MakeEmbeddingVector(train_data_preprocess[i])
    except:
        num_error += 1
    final_train_data.append(temp_train)
    final_train_label.append(temp_label)  
    
final_train_data = np.asarray(final_train_data)
final_train_label = np.asarray(final_train_label)

with open('final_train_data2.pickle', 'wb') as f:
    pickle.dump(final_train_data, f, pickle.HIGHEST_PROTOCOL)
    
with open('final_train_label2.pickle', 'wb') as f:
    pickle.dump(final_train_label, f, pickle.HIGHEST_PROTOCOL)    
    

print('number of error is : ', num_error)
print('total data is : ', len(final_train_data))
