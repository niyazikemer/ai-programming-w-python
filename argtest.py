
import argparse

parser = argparse.ArgumentParser(
    description='Some arg_pass for my project',
)
parser.add_argument(type=str, dest='data_dir')
parser.add_argument('--save_dir', dest='save_dir', type=str)
parser.add_argument('--arch', dest='arch', type=str,  default='densenet161')
parser.add_argument('--learning_rate', dest='learning_rate', type=float)
parser.add_argument('--hidden_units', dest='hidden_units', type=int)
parser.add_argument('--epochs', dest='epochs', type=int)

args = parser.parse_args()


print(args.arch)
print (int(100/3))