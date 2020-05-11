import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b",type=int)
parser.add_argument("--c",type=int)
#可选参数在输入过程中没有输入时候，自动赋值为None
parser.add_argument("-d",type=int)

parser.add_argument("-e", "--eee",type=int)#不论输入存在eee
parser.add_argument("-fff", "--f",type=int)#不论输入存在f
parser.add_argument("-ggg", "-g",type=int)#不论输入存在ggg
parser.add_argument("-hhh", "--h",type=int)#不论输入存在h
parser.add_argument("--iii", "--i",type=int)#不论输入存在iii

args = parser.parse_args('-b 100 --c 200  -e 400 -fff 500 -g 600 --h 700 --i 800'.split())
print(args)# Namespace中有变量名的信息
parser.print_usage()
parser.print_help()
