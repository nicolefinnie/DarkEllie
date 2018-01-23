#/usr/bin/python

with open("coco.names") as f:
	label_list = f.read().splitlines()
f.close()


with open("cocolabels.txt", 'w') as f:
	for label in label_list:
		print('"'+label+'",', file=f)
f.close()
