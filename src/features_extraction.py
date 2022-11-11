import cv2
import sys

def main(image_size):

	fout = open('features.txt','w')
	
	with open('../digits/files.txt', 'r') as file:
		files_names = [line.split() for line in file]
	
	for f in files_names:
		# print('reading: ' + f[0] + ' - ' + f[1])
		image = cv2.imread('../digits/' + f[0])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (20, 20))
	
		fout.write(str(f[1]) +  ' ')

		index = 0
		for i in range(int(image_size)):
			for j in range(int(image_size)):
				if(image[i][j] > 128):
					v = 0
				else:
					v = 1	
			
				fout.write(str(index) + ':' + str(v) + ' ')
				index = index + 1
		fout.write("\n")


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit('Use: knn.py <image size>')

	main(sys.argv[1])