import xlrd
import pickle
import types

def main():
	bk = xlrd.open_workbook('./RawDataset/gnd.xls')
	shxrange = range(bk.nsheets)
	sh = bk.sheet_by_name("result")
	nrows = sh.nrows
	ncols = sh.ncols

	target = {}
	for i in range(nrows):
		target[sh.cell_value(i,0)] = sh.cell_value(i,1)

	output = open('./Network/data/gndtruth.pkl', 'wb')
	pickle.dump(target, output)
	output.close()


if __name__ == '__main__':
	main()