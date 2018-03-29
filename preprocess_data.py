from random import shuffle

def main():
	with open('./data/foody/all.raw', encoding='utf-8') as f:
		revs = f.readlines()
	with open('./data/foody/all.raw', 'w', encoding='utf-8') as f:
		shuffle(revs)
		f.writelines(revs) 

if __name__ == '__main__':
	main()