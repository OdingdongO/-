import csv

csvFile_224 = open('test.csv')
#csvFile_448 = open('resnet50-448-448-0.9707_test.csv')
reader224 = csv.reader(csvFile_224)
#reader448 = csv.reader(csvFile_448)
result224 = {}
result448 = {}
for item in reader224:
	result224[item[0][:45]] = item[0][45:]


#for item in reader448:
#	result448[item[0][:45]] = item[0][45:]


#print(result448)
# a=[]
count={}
# for k,v in result224.items():
#  	count[v]=
for k,v in result224.items():
	if(v in count):
		count[v] = count[v]+1
	else:
		count[v]=1

for k,v in count.items():
	if(v!=10):
		print(k,v)







# def dict2list(dic:dict):
# 	keys = dic.keys()
# 	vals = dic.values()
# 	lst = [(key,val) for key , val in zip(keys,vals) ]
# 	return lst
# #csvFile_448.close

# print(sorted(dict2list(count),key = lambda x:x[0],reverse = True))
csvFile_224.close

