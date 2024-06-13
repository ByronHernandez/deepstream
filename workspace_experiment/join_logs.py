import os

# paths = ['triesMultipleStrings/conf2/outCommunicator_Stream0.log',
#          'triesMultipleStrings/conf2/outCommunicator_Stream1.log']
# outname = 'join_log_comm_0_1.log'

# paths = ['triesMultipleStrings/conf2/outCommunicator_Stream0.log',
#          'triesMultipleStrings/conf2/outCommunicator_Stream1.log',
#          'triesMultipleStrings/conf2/outCommunicator_Stream2.log',
#          'triesMultipleStrings/conf2/outCommunicator_Stream3.log']
# outname = 'join_log_comm_0_1_2_3.log'

paths = ['triesMultipleStrings/conf2/outMVReAssoc_Stream0.log',
         'triesMultipleStrings/conf2/outMVReAssoc_Stream1.log']
outname = 'join_log_mv_0_1.log'

# paths = ['triesMultipleStrings/conf2/outMVReAssoc_Stream0.log',
#          'triesMultipleStrings/conf2/outMVReAssoc_Stream1.log'
#          'triesMultipleStrings/conf2/outMVReAssoc_Stream2.log',
#          'triesMultipleStrings/conf2/outMVReAssoc_Stream3.log']
# outname = 'join_log_mv_0_1_2_3.log'

if  __name__ == '__main__':
    # List of log files
    lines = []
    # Read the logs
    for path in paths:
        with open(path, 'r') as f:
            lines.extend(f.readlines())
    # Sort the lines
    lines.sort()
    # Write the sorted lines to a new file
    with open(outname, 'w') as f:
        f.writelines(lines)
    