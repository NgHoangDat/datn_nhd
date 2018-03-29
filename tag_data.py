
import sys
#from six.moves import cPickle

from utils import process_vi
def main():
	file_path = sys.argv[1]
	save_file_path = sys.argv[2]
	i = int(sys.argv[3])

	revs = []
	
	with open(file_path, 'r', encoding='utf-8') as f:
		lines = f.readlines()[i:]
		for line in lines:
			try:
				i += 1
				print("=" * 80 + '\n')

				line = process_vi(line).split()

				print(" ".join(line) + '\n')
				print(" ".join([line[i] + "/" + str(i) for i in range(len(line))]) + "\n")

				if input("skip this line? y/[n]: ") == 'y':
					continue

				if input("retokenize this line? y/[n]: ") == 'y':
					join_word_list = input("Join list: ")
					join_word_list = [[int(w) for w in jw.strip().split()] for jw in join_word_list.split(',')]
					print(join_word_list)
					new_line = []
					j = 0
					k = 0
					while k < len(line):
						if j < len(join_word_list):
							if k == join_word_list[j][0]:
								next_word = "_".join([line[h] for h in join_word_list[j]])
								k += len(join_word_list[j])
								j += 1
							else:
								next_word = line[k]
								k += 1
							new_line.append(next_word)
						else:
							new_line += line[k:]
							break
					line = new_line
					print(" ".join(line) + '\n')
					print(" ".join([line[i] + "/" + str(i) for i in range(len(line))]) + "\n")							

				while True:
					print("\n")

					target_word_idx = int(input("target word idx: "))
					sentiment = input("sentiment: ")

					save_line = list(line)
					target_word, save_line[target_word_idx] = save_line[target_word_idx], '$t$'
					revs.append((i, sentiment, target_word, " ".join(save_line)))
					
					if (input("next line? y/[n]: ") == 'y'):
						break
			except EOFError:
				print(len(revs))
				with open(save_file_path, 'a+', encoding='utf-8') as ouf:
					for rev in revs:
						ouf.write("{0}\t{1}\t{2}\t{3}\n".format(rev[0], rev[1], rev[2], rev[3]))

if __name__ == '__main__':
	main()